from spacy_partial_tagger.label_indexers import configure_transformer_label_indexer


def test_transformer_label_indexer() -> None:
    padding_index = -1
    unknown_index = -100
    label_indexer = configure_transformer_label_indexer(padding_index, unknown_index)
    tag_to_id = {"O": 0, "U-LOC": 1, "B-LOC": 2, "L-LOC": 3}

    # <s> Tok yo is the capital of Japan . </s>
    tags = ["O", "B-LOC", "L-LOC", "O", "O", "O", "O", "U-LOC", "O", "O"]

    tag_indices = label_indexer([tags], tag_to_id)

    assert tag_indices == [[0, 2, 3, -100, -100, -100, -100, 1, -100, 0]]
