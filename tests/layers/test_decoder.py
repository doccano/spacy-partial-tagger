import torch
from partial_tagger.crf.decoders import ConstrainedViterbiDecoder
from partial_tagger.crf.layer import CRF

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
    crf = CRF(128, len(id_to_tag))
    decoder1 = ConstrainedDecoder(
        start_constrains, end_constrains, transition_constraints
    )
    decoder2 = ConstrainedViterbiDecoder(
        crf, start_constrains, end_constrains, transition_constraints
    )
    text_features = torch.randn(3, 11, 128)

    _, expected = decoder2(text_features)

    assert torch.allclose(decoder1(crf(text_features)), expected)
