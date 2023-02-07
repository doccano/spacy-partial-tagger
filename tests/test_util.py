import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from spacy_partial_tagger.util import get_alignments


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )


def test_get_alignments(tokenizer: PreTrainedTokenizer) -> None:
    text = "日本基督教団阿佐ヶ谷教会の牧師を務める。"
    alignment = get_alignments(tokenizer, text, tokenizer(text).input_ids)

    assert alignment == [
        [],
        [0, 1],
        [2, 3, 4],
        [5],
        [6, 7],
        [8, 9],
        [10, 11],
        [12],
        [13, 14],
        [15],
        [16, 17, 18],
        [19],
        [],
    ]


def test_get_alignments_handles_unknown_tokens(tokenizer: PreTrainedTokenizer) -> None:
    text = "武帝 の 時 に 貮師 将軍 李 広利 の 仮 司馬 と なっ て 匈奴 攻め に 従軍 し た 。"
    alignment = get_alignments(tokenizer, text, tokenizer(text).input_ids)

    assert alignment == [
        [],
        [0],
        [1],
        [3],
        [5],
        [7],
        [9, 10],
        [12, 13],
        [15],
        [17],
        [18],
        [20],
        [22],
        [24, 25],
        [27],
        [29, 30],
        [32],
        [34, 35],
        [37, 38],
        [40],
        [42, 43],
        [45],
        [47],
        [49],
        [],
    ]
