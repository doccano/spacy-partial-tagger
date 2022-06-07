from typing import Any, List, Optional, Tuple, cast

import torch
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import ArgsKwargs, Model, chain, torch2xp, with_getitem, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import Floats2d, Floats3d, Floats4d, Ints1d, Ints2d

from .aligners import Aligner
from .layers.crf import CRF
from .layers.decoder import ConstrainedDecoder, get_constraints
from .restructure import with_restructure


@registry.architectures.register("spacy-partial-tagger.PartialTagger.v2")
def build_partial_tagger_v2(
    misaligned_tok2vec: Model[List[Doc], Tuple[List[Floats2d], List[Aligner]]],
    crf: Model[Tuple[List[Floats2d], Ints1d], Floats4d],
    decoder: Model[Tuple[Floats4d, Ints1d], Ints2d],
) -> Model:
    return Model(
        name="partial_tagger",
        forward=forward,
        init=init,
        layers=[misaligned_tok2vec, crf, decoder],
    )


def forward(model: Model, X: Any, is_train: bool) -> tuple:
    (embeddings, aligners), backward1 = model.layers[0](X[0], is_train)
    log_potentials, backward2 = model.layers[1]([embeddings, X[1]], is_train)
    tag_indices, _ = model.layers[2]([log_potentials, X[1]], is_train)

    def backward(dY: tuple) -> None:
        d_embeddings, _ = backward2(dY)
        backward1([d_embeddings, None])

    return ((log_potentials, tag_indices), aligners), backward


def init(model: Model, X: Any = None, Y: Any = None) -> None:
    if Y is not None:
        Y = Y[0]
    for layer in model.layers:
        layer.initialize(X, Y)


@registry.architectures.register("spacy-partial-tagger.PartialTagger.v1")
def build_partial_tagger(
    misaligned_tok2vec: Model[List[Doc], Tuple[List[Floats2d], List[Aligner]]],
    nI: int,
    nO: Optional[int] = None,
    *,
    dropout: float = 0.2,
    padding_index: int = -1,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None
) -> Model:

    partial_tagger: Model = with_getitem(
        0,
        Model(
            name="partial_tagger",
            forward=partial_tagger_forward,
            init=partial_tagger_init,
            dims={"nI": nI, "nO": nO},
            attrs={
                "dropout": dropout,
                "padding_index": padding_index,
                "mixed_precision": mixed_precision,
                "grad_scaler": grad_scaler,
            },
        ),
    )
    # TODO: Get rid of this hack.
    # with_getitem doesn't correctly initialize the given layer.
    # Because it fails to handle dimensions, so I use this hack.
    partial_tagger._dims = {"nI": nI, "nO": nO}

    model: Model = chain(
        cast(
            Model[
                Tuple[List[Doc], Ints1d],
                Tuple[Tuple[List[Floats2d], List[Aligner]], Ints1d],
            ],
            with_getitem(0, misaligned_tok2vec),
        ),
        with_restructure(),
        cast(
            Model[
                Tuple[Tuple[List[Floats2d], Ints1d], List[Aligner]],
                Tuple[Tuple[Floats4d, Floats2d], List[Aligner]],
            ],
            partial_tagger,
        ),
    )
    return model


def partial_tagger_init(model: Model, X: Any = None, Y: Any = None) -> None:
    if model.layers:
        return
    if Y is None:
        Y = {0: "O"}
    if model.has_dim("nO") is None:
        model.set_dim("nO", len(Y))

    PyTorchWrapper = registry.get("layers", "PyTorchWrapper.v2")

    dropout = model.attrs["dropout"]
    padding_index = model.attrs["padding_index"]
    mixed_precision = model.attrs["mixed_precision"]
    grad_scaler = model.attrs["grad_scaler"]

    crf = PyTorchWrapper(
        CRF(model.get_dim("nI"), model.get_dim("nO"), dropout),
        convert_inputs=convert_crf_inputs,
        mixed_precision=mixed_precision,
        grad_scaler=grad_scaler,
    )
    decoder = PyTorchWrapper(
        ConstrainedDecoder(*get_constraints(Y), padding_index=padding_index),
        mixed_precision=mixed_precision,
        convert_inputs=convert_decoder_inputs,
        convert_outputs=convert_decoder_outputs,
        grad_scaler=grad_scaler,
    )

    model._layers = [crf, decoder]
    model.set_ref("crf", crf)
    model.set_ref("decoder", decoder)


def partial_tagger_forward(
    model: Model, X: Tuple[List[Floats2d], Ints1d], is_train: bool
) -> tuple:
    log_potentials, backward = model.get_ref("crf")(X, is_train)
    tag_indices, _ = model.get_ref("decoder")((log_potentials, X[1]), is_train)

    return (log_potentials, tag_indices), backward


def convert_crf_inputs(
    model: Model, X_lengths: Tuple[List[Floats2d], Ints1d], is_train: bool
) -> tuple:
    pad = model.ops.pad
    unpad = model.ops.unpad

    X, L = X_lengths

    Xt = xp2torch(pad(X), requires_grad=is_train)
    Lt = xp2torch(L)

    def convert_from_torch_backward(
        d_inputs: ArgsKwargs,
    ) -> Tuple[List[Floats2d], Ints1d]:
        dX = cast(Floats3d, torch2xp(d_inputs.args[0]))
        return cast(List[Floats2d], unpad(dX, L.tolist())), L  # type:ignore

    output = ArgsKwargs(args=(Xt, Lt), kwargs={})

    return output, convert_from_torch_backward


def convert_decoder_inputs(
    model: Model, X_lengths: Tuple[Floats4d, Ints1d], is_train: bool
) -> tuple:
    X, L = X_lengths

    Xt = xp2torch(X, requires_grad=True)
    Lt = xp2torch(L, requires_grad=False)
    output = ArgsKwargs(args=(Xt, Lt), kwargs={})
    return output, lambda d_inputs: []


def convert_decoder_outputs(
    model: Model,
    inputs_outputs: Tuple[Tuple[Floats4d, Ints1d], torch.Tensor],
    is_train: bool,
) -> tuple:
    _, Y_t = inputs_outputs
    Y = cast(Ints2d, torch2xp(Y_t))
    return Y, lambda dY: []
