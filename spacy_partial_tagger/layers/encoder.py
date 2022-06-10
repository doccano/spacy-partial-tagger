from typing import Any, Callable, List, Optional, Tuple, cast

import torch
from partial_tagger.encoders.linear import LinearCRFEncoder as OriginalLinearCRFEncoder
from spacy.util import registry
from thinc.api import ArgsKwargs, Model, torch2xp, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import Floats2d, Floats3d, Floats4d, Ints1d
from torch.nn import Dropout

from .util import get_mask


class LinearCRFEncoder(OriginalLinearCRFEncoder):
    def __init__(self, embedding_size: int, num_tags: int, dropout: float) -> None:
        super(LinearCRFEncoder, self).__init__(embedding_size, num_tags)

        self.dropout = Dropout(dropout)

    def forward(
        self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return super(LinearCRFEncoder, self).forward(self.dropout(embeddings), mask)


@registry.architectures.register("spacy-partial-tagger.LinearCRFEncoder.v1")
def build_linear_crf_encoder_v1(
    nI: int,
    nO: Optional[int] = None,
    dropout: float = 0.0,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None,
) -> Model[Tuple[List[Floats2d], Ints1d], Floats4d]:
    return Model(
        name="linear_crf_encoder",
        forward=forward,
        init=init,
        dims={"nI": nI, "nO": nO},
        attrs={
            "dropout": dropout,
            "mixed_precision": mixed_precision,
            "grad_scaler": grad_scaler,
        },
    )


def forward(
    model: Model[Tuple[List[Floats2d], Ints1d], Floats4d],
    X: Tuple[List[Floats2d], Ints1d],
    is_train: bool,
) -> Tuple[Floats4d, Callable]:

    return model.layers[0](X, is_train)


def init(model: Model, X: Any = None, Y: Any = None) -> None:
    if model.layers:
        return

    if Y is None:
        Y = {0: "O"}
    if model.has_dim("nO") is None:
        model.set_dim("nO", len(Y))

    PyTorchWrapper = registry.get("layers", "PyTorchWrapper.v2")

    dropout = model.attrs["dropout"]
    mixed_precision = model.attrs["mixed_precision"]
    grad_scaler = model.attrs["grad_scaler"]

    encoder = PyTorchWrapper(
        LinearCRFEncoder(model.get_dim("nI"), model.get_dim("nO"), dropout),
        convert_inputs=convert_inputs,
        mixed_precision=mixed_precision,
        grad_scaler=grad_scaler,
    )

    model._layers = [encoder]


def convert_inputs(
    model: Model, X_lengths: Tuple[List[Floats2d], Ints1d], is_train: bool
) -> tuple:
    pad = model.ops.pad
    unpad = model.ops.unpad

    X, L = X_lengths

    Xt = xp2torch(pad(X), requires_grad=is_train)
    Lt = xp2torch(L)
    mask = get_mask(Lt, Xt.size(1), Xt.device)

    def convert_from_torch_backward(
        d_inputs: ArgsKwargs,
    ) -> Tuple[List[Floats2d], Ints1d]:
        dX = cast(Floats3d, torch2xp(d_inputs.args[0]))
        return cast(List[Floats2d], unpad(dX, L.tolist())), L  # type:ignore

    output = ArgsKwargs(args=(Xt, mask), kwargs={})

    return output, convert_from_torch_backward
