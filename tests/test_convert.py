from spacy_partial_tagger.convert import converter


def test_converter() -> None:
    tokens = ["Tim", "Cook", "is", "the", "CEO", "of", "Apple", "."]
    annotations = [
        {"start": 0, "end": 2, "type": "PER"},
        {"start": 6, "end": 7, "type": "ORG"},
    ]
    char_tokens, char_annotations = converter(tokens, annotations)

    assert char_tokens == [
        "T",
        "i",
        "m",
        " ",
        "C",
        "o",
        "o",
        "k",
        " ",
        "i",
        "s",
        " ",
        "t",
        "h",
        "e",
        " ",
        "C",
        "E",
        "O",
        " ",
        "o",
        "f",
        " ",
        "A",
        "p",
        "p",
        "l",
        "e",
        " ",
        ".",
    ]

    assert char_annotations == [(0, 8, "PER"), (23, 28, "ORG")]
