from spacy_partial_tagger.aligners import (
    PassThroughAligner,
    TransformerAligner,
    convert_tags,
)


def test_pass_through_aligner() -> None:
    tags = ["B-X", "L-X", "O", "U-Y", "O", "O"]
    aligner = PassThroughAligner()

    assert aligner.to_subword(tags) == tags
    assert aligner.from_subword(tags) == tags


def test_convert_tags() -> None:
    char_offsets_by_subword = [
        (0, 0),
        (0, 3),
        (3, 5),
        (6, 8),
        (9, 12),
        (13, 20),
        (21, 23),
        (24, 29),
        (29, 30),
        (0, 0),
    ]
    char_offsets_by_token = [
        (0, 5),
        (6, 8),
        (9, 12),
        (13, 20),
        (21, 23),
        (24, 29),
        (29, 30),
    ]
    char_length = 30
    token_length = 7
    subword_length = 10

    tags_token_base = ["U-LOC", "O", "O", "O", "O", "U-LOC", "O"]
    tags_subword_base = ["O", "B-LOC", "L-LOC", "O", "O", "O", "O", "U-LOC", "O", "O"]

    assert (
        convert_tags(
            tags_token_base,
            char_offsets_by_token,
            char_offsets_by_subword,
            char_length,
            subword_length,
        )
        == tags_subword_base
    )

    assert (
        convert_tags(
            tags_subword_base,
            char_offsets_by_subword,
            char_offsets_by_token,
            char_length,
            token_length,
        )
        == tags_token_base
    )


def test_transformer_aligner() -> None:
    # Tokyo is the capital of Japan.
    tags = [
        "B-X",  # T
        "I-X",  # o
        "I-X",  # k
        "I-X",  # y
        "L-X",  # o
        "O",
        "O",  # i
        "O",  # s
        "O",
        "O",  # t
        "O",  # h
        "O",  # e
        "O",
        "O",  # c
        "O",  # a
        "O",  # p
        "O",  # i
        "O",  # t
        "O",  # a
        "O",  # l
        "O",
        "O",  # o
        "O",  # f
        "O",
        "B-X",  # J
        "I-X",  # a
        "I-X",  # p
        "I-X",  # a
        "L-X",  # n
        "O",  # .
    ]
    # '<s>', 'Tok', 'yo', 'Ġis', 'Ġthe', 'Ġcapital', 'Ġof', 'ĠJapan', '.', '</s>'
    tags_subword = ["O", "B-X", "L-X", "O", "O", "O", "O", "U-X", "O", "O"]
    aligner = TransformerAligner(
        [
            [],
            [0, 1, 2],
            [3, 4],
            [6, 7],
            [9, 10, 11],
            [13, 14, 15, 16, 17, 18, 19],
            [21, 22],
            [24, 25, 26, 27, 28],
            [29],
            [],
        ]
    )

    assert aligner.to_subword(tags) == tags_subword
    assert aligner.from_subword(tags_subword) == tags
