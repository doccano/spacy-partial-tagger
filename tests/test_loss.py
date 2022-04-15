import numpy as np

from spacy_partial_tagger.loss import ExpectedEntityRatioLoss


def test_expected_entity_ratio_loss_returns_expected_shape() -> None:
    log_potentials = np.random.randn(3, 7, 5, 5)
    tag_indices = np.array(
        [
            [0, 1, -100, 3, 4, -1, -1],
            [-100, -100, -1, -1, -1, -1, -1],
            [0, 1, 2, 3, 4, 0, 1],
        ]
    )
    padding_index = -1
    unknown_index = -100
    outside_index = 0
    loss_func = ExpectedEntityRatioLoss(padding_index, unknown_index, outside_index)

    grad, loss = loss_func(log_potentials, tag_indices)  # type:ignore

    assert grad.shape == (3, 7, 5, 5)
    assert loss.shape == tuple()
