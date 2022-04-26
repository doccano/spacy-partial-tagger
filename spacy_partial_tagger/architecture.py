from typing import Any, List, Optional, Tuple, cast

import torch
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import (
    ArgsKwargs,
    Model,
    chain,
    list2array,
    torch2xp,
    with_getitem,
    xp2torch,
)
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import Floats2d, Floats3d, Floats4d, Ints1d, Ints2d

from spacy_partial_tagger.layers.crf import CRF
from spacy_partial_tagger.layers.decoder import ConstrainedDecoder, get_constraints


@registry.architectures.register("spacy-partial-tagger.PartialTagger.v1")
def build_partial_tagger(
    tok2vec: Model[List[Doc], List[Floats2d]],
    nI: int,
    nO: Optional[int] = None,
    *,
    dropout: float = 0.2,
    padding_index: int = -1,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None
) -> Model:

    partial_tagger: Model = Model(
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
    )

    model: Model = chain(
        cast(
            Model[Tuple[List[Doc], Ints1d], Tuple[Floats2d, Ints1d]],
            with_getitem(
                0, chain(tok2vec, cast(Model[List[Floats2d], Floats2d], list2array()))
            ),
        ),
        partial_tagger,
    )
    model.set_ref("partial_tagger", partial_tagger)
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
        convert_outputs=convert_crf_outputs,
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
    model: Model, X: Tuple[Floats2d, Ints1d], is_train: bool
) -> tuple:
    log_potentials, backward = model.get_ref("crf")(X, is_train)

    tag_indices, _ = model.get_ref("decoder")((log_potentials, X[1]), is_train)

    return (log_potentials, tag_indices), backward


def convert_crf_inputs(
    model: Model, X_lengths: Tuple[Floats2d, Ints1d], is_train: bool
) -> tuple:
    flatten = model.ops.flatten
    unflatten = model.ops.unflatten
    pad = model.ops.pad
    unpad = model.ops.unpad

    X, L = X_lengths

    Xt = xp2torch(pad(unflatten(X, L)), requires_grad=is_train)
    Lt = xp2torch(L)

    def convert_from_torch_backward(d_inputs: ArgsKwargs) -> Tuple[Floats2d, Ints1d]:
        dX = cast(Floats3d, torch2xp(d_inputs.args[0]))
        return cast(Floats2d, flatten(unpad(dX, L.tolist()))), L  # type:ignore

    output = ArgsKwargs(args=(Xt, Lt), kwargs={})

    return output, convert_from_torch_backward


def convert_crf_outputs(model: Model, inputs_outputs: tuple, is_train: bool) -> tuple:

    _, Y_t = inputs_outputs

    def convert_for_torch_backward(dY: Floats4d) -> ArgsKwargs:
        dY_t = xp2torch(dY)
        return ArgsKwargs(args=([Y_t],), kwargs={"grad_tensors": dY_t})

    Y = cast(Floats4d, torch2xp(Y_t))
    return Y, convert_for_torch_backward


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
