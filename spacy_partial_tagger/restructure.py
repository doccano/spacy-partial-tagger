from typing import Any, Callable, List, Tuple

from thinc.api import Model
from thinc.types import Floats2d, Ints1d

from .aligners import Aligner


def with_restructure() -> Model:
    return Model(name="with_restructure", forward=forward)


def forward(
    model: Model, X: Any, is_train: bool
) -> Tuple[Tuple[Tuple[List[Floats2d], Ints1d], List[Aligner]], Callable]:
    (vec, align), lengths = X

    def backward(dY: tuple) -> tuple:
        (dvec, dlengths), dalign = dY
        return (dvec, dalign), dlengths

    return ((vec, lengths), align), backward
