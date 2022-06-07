import torch
from partial_tagger.decoders.viterbi import ConstrainedViterbiDecoder
from partial_tagger.encoders.linear import LinearCRFEncoder

from spacy_partial_tagger.layers.decoder import ConstrainedDecoder, get_constraints


def test_constrained_decoder_returns_same_tensor_as_partial_tagger() -> None:
    id_to_tag = {
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
    start_constrains, end_constrains, transition_constraints = get_constraints(
        id_to_tag
    )
    crf = LinearCRFEncoder(128, len(id_to_tag))
    decoder1 = ConstrainedDecoder(
        start_constrains, end_constrains, transition_constraints
    )
    decoder2 = ConstrainedViterbiDecoder(
        start_constrains, end_constrains, transition_constraints
    )
    embeddings = torch.randn(3, 11, 128)
    log_potentials = crf(embeddings)

    assert torch.allclose(decoder1(log_potentials), decoder2(log_potentials))
