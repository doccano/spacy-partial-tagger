import spacy
from spacy.tokens import Doc

from spacy_partial_tagger.util import get_alignments, make_char_based_doc


def test_get_alignments_handles_unknown_tokens() -> None:
    text = "武帝 の 時 に 貮師 将軍 李 広利 の 仮 司馬 と なっ て 匈奴 攻め に 従軍 し た 。"
    subwords = [
        "武",
        "帝",
        "の",
        "時",
        "に",
        "[UNK]",
        "師",
        "将",
        "軍",
        "李",
        "広",
        "利",
        "の",
        "仮",
        "司",
        "馬",
        "と",
        "な",
        "っ",
        "て",
        "匈",
        "奴",
        "攻",
        "め",
        "に",
        "従",
        "軍",
        "し",
        "た",
        "。",
    ]
    alignment = get_alignments(list(text), subwords)

    assert alignment == [
        [0],
        [1],
        [3],
        [5],
        [7],
        [9],
        [10],
        [12],
        [13],
        [15],
        [17],
        [18],
        [20],
        [22],
        [24],
        [25],
        [27],
        [29],
        [30],
        [32],
        [34],
        [35],
        [37],
        [38],
        [40],
        [42],
        [43],
        [45],
        [47],
        [49],
    ]


def test_make_char_based_doc() -> None:
    nlp = spacy.blank("en")
    words = "Tokyo is the capital of Japan .".split()
    tags = ["U-LOC", "O", "O", "O", "O", "U-LOC", "O"]
    spaces = [True] * len(words)
    spaces[-2:] = [False] * 2
    doc = Doc(nlp.vocab, words=words, spaces=spaces)

    char_doc = make_char_based_doc(doc, tags)

    assert char_doc.text == "Tokyo is the capital of Japan."
    assert [ent.text for ent in char_doc.ents] == ["Tokyo", "Japan"]
