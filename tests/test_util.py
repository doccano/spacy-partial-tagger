from spacy_partial_tagger.util import from_subword_tags, to_subword_tags


def test_to_subword_tags() -> None:
    offset_mapping = [(0, 0), (0, 3), (4, 8), (9, 13), (14, 19), (20, 21), (0, 0)]
    tags = [
        "B-PER",
        "I-PER",
        "I-PER",
        "I-PER",
        "I-PER",
        "I-PER",
        "I-PER",
        "L-PER",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
    ]

    assert to_subword_tags(tags, offset_mapping) == [
        "O",
        "B-PER",
        "L-PER",
        "O",
        "O",
        "O",
        "O",
    ]


def test_from_subword_tags() -> None:
    offset_mapping = [(0, 0), (0, 3), (4, 8), (9, 13), (14, 19), (20, 21), (0, 0)]
    subword_tags = [
        "O",
        "B-PER",
        "L-PER",
        "O",
        "O",
        "O",
        "O",
    ]

    assert from_subword_tags(subword_tags, offset_mapping, 21) == [
        "B-PER",
        "I-PER",
        "I-PER",
        "I-PER",
        "I-PER",
        "I-PER",
        "I-PER",
        "L-PER",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
    ]
