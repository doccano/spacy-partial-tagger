import pytest
from spacy import Language

from spacy_partial_tagger.tokenizer import TransformerTokenizer


@pytest.fixture
def tokenizer(nlp: Language) -> TransformerTokenizer:
    return TransformerTokenizer(nlp.vocab, "distilroberta-base")


def test_transformer_tokenizer(tokenizer: TransformerTokenizer) -> None:
    doc = tokenizer("Tokyo is the capital of Japan.")
    expected = [
        "<s>",
        "Tok",
        "yo",
        "Ġis",
        "Ġthe",
        "Ġcapital",
        "Ġof",
        "ĠJapan",
        ".",
        "</s>",
    ]

    assert [token.text for token in doc] == expected
