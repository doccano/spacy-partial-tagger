from typing import Any

from thinc.api import Model


def with_restructure() -> Model:
    return Model(name="with_restructure", forward=forward)


def forward(model: Model, X: Any, is_train: bool) -> tuple:
    (vec, align), lengths = X

    def backward(dY: tuple) -> tuple:
        (dvec, dlengths), dalign = dY
        return (dvec, dalign), dlengths

    return ((vec, lengths), align), backward
