from typing import List

import pytest
from spacy import Language
from spacy.tokens import Doc, Span

from spacy_partial_tagger.aligners import PassThroughAligner
from spacy_partial_tagger.label_indexers import configure_transformer_label_indexer


@pytest.fixture
def docs(nlp: Language) -> List[Doc]:
    words = ["<s>", "Tokyo", "is", "the", "capital", "of", "Japan", ".", "</s>"]
    doc = Doc(nlp.vocab, words=words, spaces=[True] * (len(words) - 1) + [False])
    doc.set_ents([Span(doc, 1, 2, "LOC"), Span(doc, 6, 7, "LOC")])
    return [doc]


def test_roberta_label_indexer(docs: List[Doc]) -> None:
    padding_index = -1
    unknown_index = -100
    label_indexer = configure_transformer_label_indexer(padding_index, unknown_index)
    tag_to_id = {"O": 0, "U-LOC": 1}

    tag_indices = label_indexer(docs, tag_to_id, [PassThroughAligner()] * len(docs))

    assert tag_indices == [[0, 1, -100, -100, -100, -100, 1, -100, 0]]
