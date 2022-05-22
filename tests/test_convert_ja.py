import pytest

from spacy_partial_tagger.convert_ja import tokenize


@pytest.mark.local
def test_tokenize() -> None:
    text = "外国人参政権に対する考え方の違い。"
    expected = ["外国", "人", "参政", "権", "に", "対する", "考え", "方", "の", "違い", "。"]

    assert tokenize(text) == expected
