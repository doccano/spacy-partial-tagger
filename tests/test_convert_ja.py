import pytest

from spacy_partial_tagger.convert_ja import tokenize_jumanpp


@pytest.mark.local
def test_tokenize_returns_words_list() -> None:
    text = "外国人参政権に対する考え方の違い。"
    expected = ["外国", "人", "参政", "権", "に", "対する", "考え", "方", "の", "違い", "。"]

    assert tokenize_jumanpp(text) == expected


@pytest.mark.local
def test_tokenize_skips_whitespace_if_included() -> None:
    text = "ＴＨＥ\u3000ＦＩＲＳＴ\u3000ＴＡＫＥ"
    expected = ["ＴＨＥ", "ＦＩＲＳＴ", "ＴＡＫＥ"]

    assert tokenize_jumanpp(text) == expected
