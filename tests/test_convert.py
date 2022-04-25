import pytest
from spacy import Language

from spacy_partial_tagger.convert import converter
from spacy_partial_tagger.tokenizer import TransformerTokenizer


@pytest.fixture
def transformer_tokenizer(nlp: Language) -> TransformerTokenizer:
    return TransformerTokenizer(nlp.vocab, "distilroberta-base")


def test_converter(transformer_tokenizer: TransformerTokenizer) -> None:
    tokens = ["Tokyo", "is", "the", "capital", "of", "Japan", "."]
    text = "Tokyo is the capital of Japan."
    subwords = [subword.text for subword in transformer_tokenizer(text)]
    annotations = [
        {"start": 0, "end": 1, "type": "LOC"},
        {"start": 5, "end": 6, "type": "LOC"},
    ]
    subword_annotations = converter(tokens, text, subwords, annotations)

    assert subword_annotations == [(0, 2, "LOC"), (6, 7, "LOC")]
