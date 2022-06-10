from typing import Tuple, cast

import torch
from partial_tagger.crf import functional as F
from thinc.config import registry
from thinc.loss import Loss
from thinc.types import Floats1d, Floats4d, Ints2d
from thinc.util import torch2xp, xp2torch


class ExpectedEntityRatioLoss(Loss):
    def __init__(
        self,
        padding_index: int,
        unknown_index: int,
        outside_index: int,
        expected_entity_ratio_loss_weight: float = 10.0,
        entity_ratio: float = 0.15,
        entity_ratio_margin: float = 0.05,
    ) -> None:
        super(ExpectedEntityRatioLoss, self).__init__()

        self.padding_index = padding_index
        self.unknown_index = unknown_index
        self.outside_index = outside_index

        self.expected_entity_ratio_loss_weight = expected_entity_ratio_loss_weight
        self.entity_ratio = entity_ratio
        self.entity_ratio_margin = entity_ratio_margin

    def __call__(self, guesses: Floats4d, truths: Ints2d) -> Tuple[Floats4d, Floats1d]:
        guesses_pt, truths_pt = xp2torch(guesses, requires_grad=True), xp2torch(truths)
        mask = truths_pt != self.padding_index
        truths_pt = F.to_tag_bitmap(
            truths_pt, guesses_pt.size(-1), partial_index=self.unknown_index
        )
        with torch.enable_grad():
            # log partition
            log_Z = F.forward_algorithm(guesses_pt)

            # marginal probabilities
            p = torch.autograd.grad(log_Z.sum(), guesses_pt, create_graph=True)[0].sum(
                dim=-1
            )

        p *= mask[..., None]

        expected_entity_count = (
            p[:, :, : self.outside_index].sum()
            + p[:, :, self.outside_index + 1 :].sum()
        )
        expected_entity_ratio = expected_entity_count / p.sum()
        eer_loss = torch.clamp(
            (expected_entity_ratio - self.entity_ratio).abs()
            - self.entity_ratio_margin,
            min=0,
        )

        # marginal likelihood
        score = F.multitag_sequence_score(guesses_pt, truths_pt, mask)

        loss = (
            log_Z - score
        ).mean() + self.expected_entity_ratio_loss_weight * eer_loss
        (grad,) = torch.autograd.grad(loss, guesses_pt)
        return cast(Floats4d, torch2xp(grad)), cast(Floats1d, torch2xp(loss))

    def get_grad(self, guesses: Floats4d, truths: Ints2d) -> Floats4d:
        return self(guesses, truths)[0]

    def get_loss(self, guesses: Floats4d, truths: Ints2d) -> Floats1d:
        return self(guesses, truths)[1]


@registry.losses("spacy-partial-tagger.ExpectedEntityRatioLoss.v1")
def configure_ExpectedEntityRatioLoss(
    padding_index: int,
    unknown_index: int,
    outside_index: int,
    expected_entity_ratio_loss_weight: float = 10.0,
    entity_ratio: float = 0.15,
    entity_ratio_margin: float = 0.05,
) -> ExpectedEntityRatioLoss:
    return ExpectedEntityRatioLoss(
        padding_index,
        unknown_index,
        outside_index,
        expected_entity_ratio_loss_weight,
        entity_ratio,
        entity_ratio_margin,
    )
