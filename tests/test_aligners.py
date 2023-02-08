from spacy_partial_tagger.aligners import convert_tags


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
