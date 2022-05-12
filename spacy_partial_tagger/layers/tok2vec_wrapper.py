from typing import Any, Callable, List, Tuple

from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Model
from thinc.types import Floats2d

from ..aligners import PassThroughAligner


@registry.architectures.register("spacy-partial-tagger.Tok2VecWrapper.v1")
def build_tok2vec_wrapper(
    tok2vec: Model[List[Doc], List[Floats2d]]
) -> Model[List[Doc], Tuple[List[Floats2d], List[PassThroughAligner]]]:

    return Model(
        name=f"wrapper_{tok2vec.name}", forward=forward, init=init, layers=[tok2vec]
    )


def forward(
    model: Model, X: List[Doc], is_train: bool
) -> Tuple[Tuple[List[Floats2d], List[PassThroughAligner]], Callable]:
    Y, tok2vec_backward = model.layers[0](X, is_train)
    aligners = [PassThroughAligner() for _ in range(len(X))]

    def backward(dY: Tuple[List[Floats2d], List[PassThroughAligner]]) -> List[Doc]:
        return tok2vec_backward(dY[0])

    return (Y, aligners), backward


def init(model: Model, X: Any = None, Y: Any = None) -> Model:
    model.layers[0].initialize(X=X, Y=Y)
    return model
