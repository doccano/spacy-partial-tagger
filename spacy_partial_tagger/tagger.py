from typing import Any, Callable, List, Tuple

from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Model
from thinc.types import Floats2d, Floats4d, Ints1d, Ints2d

from .aligners import Aligner
from .layers.constrainer import Constrainer


@registry.architectures.register("spacy-partial-tagger.PartialTagger.v1")
def build_partial_tagger_v1(
    misaligned_tok2vec: Model[
        List[Doc], Tuple[List[Floats2d], List[Aligner], Constrainer]
    ],
    encoder: Model[Tuple[List[Floats2d], Ints1d], Floats4d],
    decoder: Model[Tuple[Floats4d, Ints1d], Ints2d],
) -> Model[Tuple[List[Doc], Ints1d], Tuple[Floats4d, Ints2d, Aligner]]:
    return Model(
        name="partial_tagger",
        forward=forward,
        init=init,
        layers=[misaligned_tok2vec, encoder, decoder],
    )


def forward(
    model: Model[List[Doc], Tuple[Floats4d, Ints2d, Aligner]],
    X: Tuple[List[Doc], Ints1d],
    is_train: bool,
) -> Tuple[Tuple[Floats4d, Ints2d, Aligner], Callable]:

    (embeddings, aligners, constrainer, subword_lengths), backward1 = model.layers[0](
        X, is_train
    )
    log_potentials, backward2 = model.layers[1]([embeddings, subword_lengths], is_train)
    constrained_log_potentials = constrainer(model.ops, log_potentials)

    tag_indices, _ = model.layers[2](
        [constrained_log_potentials, subword_lengths], is_train
    )

    def backward(dY: Tuple[Floats4d, None, None, None]) -> dict:
        d_embeddings, _ = backward2(dY[0])
        return backward1([d_embeddings, None])

    return (log_potentials, tag_indices, aligners), backward


def init(
    model: Model[Tuple[List[Doc], Ints1d], Tuple[Floats4d, Ints2d, Aligner]],
    X: Any = None,
    Y: Any = None,
) -> None:
    for layer in model.layers:
        layer.initialize(X, Y)
