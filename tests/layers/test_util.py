import torch

from spacy_partial_tagger.layers import util


def test_get_mask() -> None:
    lengths = torch.Tensor([7, 3, 5])
    max_length = 7
    expected = torch.tensor(
        [[True] * 7, [True] * 3 + [False] * 4, [True] * 5 + [False] * 2],
    )

    assert torch.equal(util.get_mask(lengths, max_length, lengths.device), expected)
