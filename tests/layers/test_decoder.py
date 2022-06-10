from spacy_partial_tagger.layers.decoder import build_constrained_viterbi_decoder_v1


def test_constrained_viterbi_decoder() -> None:
    tags = {
        0: "O",
        1: "B-X",
        2: "I-X",
        3: "L-X",
        4: "U-X",
        5: "B-Y",
        6: "I-Y",
        7: "L-Y",
        8: "U-Y",
    }
    decoder = build_constrained_viterbi_decoder_v1()
    decoder.initialize(Y=tags)

    log_potentials = decoder.ops.alloc4f(3, 20, len(tags), len(tags))
    lengths = decoder.ops.asarray1i([20, 20, 20])
    tag_indices, _ = decoder((log_potentials, lengths), is_train=False)

    assert tag_indices.shape == (3, 20)
