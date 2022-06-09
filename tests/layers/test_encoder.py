from spacy_partial_tagger.layers.encoder import build_linear_crf_encoder_v1


def test_linear_crf_encoder() -> None:
    embedding_size = 128
    num_tags = 5
    encoder = build_linear_crf_encoder_v1(embedding_size, num_tags)
    encoder.initialize()

    batch_size = 3
    sequence_length = 20
    embeddings = [
        encoder.ops.alloc2f(sequence_length, embedding_size) for _ in range(batch_size)
    ]
    lengths = encoder.ops.asarray1i([sequence_length] * batch_size)

    log_potentials, _ = encoder((embeddings, lengths), is_train=False)

    assert log_potentials.shape == (batch_size, sequence_length, num_tags, num_tags)
