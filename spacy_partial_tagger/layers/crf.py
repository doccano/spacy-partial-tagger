from typing import Any, Callable, List, Optional, Tuple, cast

import torch
from partial_tagger.encoders.linear import LinearCRFEncoder
from spacy.util import registry
from thinc.api import ArgsKwargs, Model, torch2xp, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import Floats2d, Floats3d, Floats4d, Ints1d
from torch import nn


def get_mask(
    lengths: torch.Tensor, max_length: int, device: torch.device
) -> torch.Tensor:
    return (
        torch.arange(
            max_length,
            device=device,
        )[None, :]
        < lengths[:, None]
    )


@registry.architectures.register("spacy-partial-tagger.CRF.v1")
def build_crf_v1(
    nI: int,
    nO: Optional[int] = None,
    dropout: float = 0.0,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None,
) -> Model[Tuple[List[Floats2d], Ints1d], Floats4d]:
    return Model(
        name="crf",
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

    dropouted, backward1 = model.layers[0](X, is_train)
    log_potentials, backward2 = model.layers[1](X, is_train)

    def backward(dY: Floats4d) -> List[Floats2d]:
        return backward1(backward2(dY))

    return log_potentials, backward


def init(model: Model, X: Any = None, Y: Any = None) -> None:
    if model.layers:
        return

    if Y is None:
        Y = {0: "O"}
    if model.has_dim("nO") is None:
        model.set_dim("nO", len(Y))

    PyTorchWrapper = registry.get("layers", "PyTorchWrapper.v2")
    Dropout = registry.get("layers", "Dropout.v1")

    dropout = model.attrs["dropout"]
    mixed_precision = model.attrs["mixed_precision"]
    grad_scaler = model.attrs["grad_scaler"]

    crf = PyTorchWrapper(
        LinearCRFEncoder(model.get_dim("nI"), model.get_dim("nO")),
        convert_inputs=convert_inputs,
        mixed_precision=mixed_precision,
        grad_scaler=grad_scaler,
    )

    model._layers = [Dropout(dropout), crf]


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


class CRF(nn.Module):
    def __init__(
        self, embedding_size: int, num_tags: int, dropout: float = 0.2
    ) -> None:
        super(CRF, self).__init__()

        self.crf = LinearCRFEncoder(embedding_size, num_tags)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, embeddings: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is None:
            mask = embeddings.new_ones(embeddings.shape[:-1], dtype=torch.bool)
        else:
            mask = (
                torch.arange(
                    embeddings.size(1),
                    device=embeddings.device,
                )[None, :]
                < lengths[:, None]
            )
        return self.crf(self.dropout(embeddings), mask)
