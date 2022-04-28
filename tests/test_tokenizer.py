import pytest
from spacy import Language

from spacy_partial_tagger.tokenizer import CharacterTokenizer, TransformerTokenizer


@pytest.fixture
def character_tokenizer(nlp: Language) -> CharacterTokenizer:
    return CharacterTokenizer(nlp.vocab)


@pytest.fixture
def transformer_tokenizer(nlp: Language) -> TransformerTokenizer:
    return TransformerTokenizer(nlp.vocab, "distilroberta-base")


def test_character_tokenizer(character_tokenizer: CharacterTokenizer) -> None:
    text = "Tokyo is the capital of Japan."
    doc = character_tokenizer(text)
    expected = list(text)

    assert [token.text for token in doc] == expected


def test_transformer_tokenizer(transformer_tokenizer: TransformerTokenizer) -> None:
    text = "Tokyo is the capital of Japan."
    doc = transformer_tokenizer(text)
    expected = [
        "<s>",
        "Tok",
        "yo",
        " is",
        " the",
        " capital",
        " of",
        " Japan",
        ".",
        "</s>",
    ]

    assert [token.text for token in doc] == expected
    assert doc.text == "<s>" + text + "</s>"
