from typing import Any, Callable, Optional, Tuple, cast

import torch
from partial_tagger.decoders.viterbi import ViterbiDecoder
from spacy.util import registry
from thinc.api import ArgsKwargs, Model, torch2xp, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import Floats4d, Ints1d, Ints2d

from .util import get_mask


@registry.architectures.register("spacy-partial-tagger.ViterbiDecoder.v1")
def build_viterbi_decoder_v1(
    padding_index: int = -1,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None,
) -> Model[Tuple[Floats4d, Ints1d], Ints2d]:
    return Model(
        name="viterbi_decoder",
        forward=forward,
        init=init,
        attrs={
            "padding_index": padding_index,
            "mixed_precision": mixed_precision,
            "grad_scaler": grad_scaler,
        },
    )


def forward(
    model: Model[Tuple[Floats4d, Ints1d], Ints2d],
    X: Tuple[Floats4d, Ints1d],
    is_train: bool,
) -> Tuple[Ints2d, Callable]:
    return model.layers[0](X, is_train)


def init(
    model: Model[Tuple[Floats4d, Ints1d], Ints2d], X: Any = None, Y: Any = None
) -> None:
    if model.layers:
        return

    if Y is None:
        Y = {0: "O"}

    PyTorchWrapper = registry.get("layers", "PyTorchWrapper.v2")

    padding_index = model.attrs["padding_index"]
    mixed_precision = model.attrs["mixed_precision"]
    grad_scaler = model.attrs["grad_scaler"]
    decoder = PyTorchWrapper(
        ViterbiDecoder(padding_index=padding_index),
        mixed_precision=mixed_precision,
        convert_inputs=convert_inputs,
        convert_outputs=convert_outputs,
        grad_scaler=grad_scaler,
    )

    model._layers = [decoder]


def convert_inputs(
    model: Model[Tuple[Floats4d, Ints1d], Ints2d],
    X_lengths: Tuple[Floats4d, Ints1d],
    is_train: bool,
) -> Tuple[ArgsKwargs, Callable]:
    X, L = X_lengths

    Xt = xp2torch(X, requires_grad=True)
    Lt = xp2torch(L, requires_grad=False)
    mask = get_mask(Lt, Xt.size(1), Xt.device)
    output = ArgsKwargs(args=(Xt, mask), kwargs={})
    return output, lambda d_inputs: []


def convert_outputs(
    model: Model[Tuple[Floats4d, Ints1d], Ints2d],
    inputs_outputs: Tuple[Tuple[Floats4d, Ints1d], torch.Tensor],
    is_train: bool,
) -> Tuple[Ints2d, Callable]:
    _, Y_t = inputs_outputs
    Y = cast(Ints2d, torch2xp(Y_t))
    return Y, lambda dY: []
