import pytest
import spacy


@pytest.fixture
def nlp() -> spacy.Language:
    return spacy.blank("en")
