from spacy.language import Language
from spacy.tokens import Doc

from spacy_partial_tagger.layers.decoder import build_viterbi_decoder_v1
from spacy_partial_tagger.layers.encoder import build_linear_crf_encoder_v1
from spacy_partial_tagger.layers.tok2vec_transformer import (
    build_misaligned_tok2vec_transformer,
)
from spacy_partial_tagger.tagger import build_partial_tagger_v1


def test_partial_tagger(nlp: Language) -> None:
    tags = {
        0: "O",
        1: "B-X",
        2: "I-X",
        3: "L-X",
        4: "U-X",
        5: "B-Y",
        6: "I-Y",
        7: "L-Y",
        8: "U-Y",
    }
    tagger = build_partial_tagger_v1(
        build_misaligned_tok2vec_transformer("distilroberta-base"),
        build_linear_crf_encoder_v1(768),
        build_viterbi_decoder_v1(-1),
    )
    tagger.initialize(Y=tags)

    text = "Tokyo is the capital of Japan."
    docs = [
        Doc(
            nlp.vocab,
            words=list(text),
            spaces=[False] * len(text),
        )
    ]

    (log_potentials, tag_indices, _), _ = tagger(docs, is_train=False)

    # 10 is the length of sub-words of text.
    assert log_potentials.shape == (1, 10, len(tags), len(tags))
    assert tag_indices.shape == (1, 10)
