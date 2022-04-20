import pytest
from spacy import Language

from spacy_partial_tagger.tokenizer import CharacterTokenizer


@pytest.fixture
def character_tokenizer(nlp: Language) -> CharacterTokenizer:
    return CharacterTokenizer(nlp.vocab)


def test_character_tokenizer(character_tokenizer: CharacterTokenizer) -> None:
    text = "Tokyo is the capital of Japan."
    doc = character_tokenizer(text)
    expected = list(text)

    assert [token.text for token in doc] == expected
