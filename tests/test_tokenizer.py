import pytest
from spacy import Language

from spacy_partial_tagger.tokenizer import CharacterTokenizer, TransformerTokenizer


@pytest.fixture
def transformer_tokenizer(nlp: Language) -> TransformerTokenizer:
    return TransformerTokenizer(nlp.vocab, "distilroberta-base")


@pytest.fixture
def character_tokenizer(nlp: Language) -> CharacterTokenizer:
    return CharacterTokenizer(nlp.vocab)


def test_transformer_tokenizer(transformer_tokenizer: TransformerTokenizer) -> None:
    doc = transformer_tokenizer("Tokyo is the capital of Japan.")
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


def test_character_tokenizer(character_tokenizer: CharacterTokenizer) -> None:
    text = "Tokyo is the capital of Japan."
    doc = character_tokenizer(text)
    expected = list(text)

    assert [token.text for token in doc] == expected
