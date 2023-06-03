from partial_tagger.data import LabelSet
from spacy.language import Language
from spacy.tokens import Doc

from spacy_partial_tagger.tagger import build_partial_tagger_v1


def test_partial_tagger(nlp: Language) -> None:
    tagger = build_partial_tagger_v1("distilroberta-base", -1)
    label_set = LabelSet({"X", "Y"})
    tagger.initialize(Y=label_set)

    text = "Tokyo is the capital of Japan."
    docs = [
        Doc(
            nlp.vocab,
            words=list(text),
            spaces=[False] * len(text),
        )
    ]

    (log_potentials, tag_indices), _ = tagger(docs, is_train=False)

    # 10 is the length of sub-words of text.
    num_tags = label_set.get_tag_size()
    assert log_potentials.shape == (1, 10, num_tags, num_tags)
    assert tag_indices.shape == (1, 10)
