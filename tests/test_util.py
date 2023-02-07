import spacy
from spacy.tokens import Doc
from transformers import AutoTokenizer

from spacy_partial_tagger.util import get_alignments, make_char_based_doc


def test_get_alignments_handles_unknown_tokens() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )
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
