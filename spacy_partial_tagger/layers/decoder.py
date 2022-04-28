from typing import Dict, List, Optional, Tuple

import torch
from allennlp.modules.conditional_random_field import is_transition_allowed
from partial_tagger.functional.crf import amax, constrain_log_potentials
from torch import nn


class ConstrainedDecoder(nn.Module):
    def __init__(
        self,
        start_constraints: List[bool],
        end_constraints: List[bool],
        transition_constraints: List[List[bool]],
        padding_index: Optional[int] = -1,
    ) -> None:
        super(ConstrainedDecoder, self).__init__()

        self.start_constraints = nn.Parameter(
            torch.tensor(start_constraints), requires_grad=False
        )
        self.end_constraints = nn.Parameter(
            torch.tensor(end_constraints), requires_grad=False
        )
        self.transition_constraints = nn.Parameter(
            torch.tensor(transition_constraints), requires_grad=False
        )
        self.padding_index = padding_index

    def forward(
        self,
        log_potentials: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if lengths is None:
            mask = log_potentials.new_ones(log_potentials.shape[:-2], dtype=torch.bool)
        else:
            mask = (
                torch.arange(
                    log_potentials.size(1),
                    device=log_potentials.device,
                )[None, :]
                < lengths[:, None]
            )

        log_potentials.requires_grad_()

        with torch.enable_grad():
            constrained_log_potentials = constrain_log_potentials(
                log_potentials,
                mask,
                self.start_constraints,
                self.end_constraints,
                self.transition_constraints,
            )

            max_score = amax(constrained_log_potentials)

            (tag_matrix,) = torch.autograd.grad(
                max_score.sum(), constrained_log_potentials
            )
            tag_matrix = tag_matrix.long()

            tag_bitmap = tag_matrix.sum(dim=-2)

            tag_indices = tag_bitmap.argmax(dim=-1)

        tag_indices = tag_indices * mask + self.padding_index * (~mask)
        return tag_indices


def get_constraints(tag_dict: Dict[int, str]) -> Tuple[list, list, list]:
    """Computes start/end/transition constraints for a CRF.
    Args:
        tag_dict: A dictionary mapping tag_ids to tags.
    Returns:
        A tuple of start/end/transition constraints. start/end is a list of boolean
        indicating allowed tags. transition is a nested list of boolean.
    """
    num_tags = len(tag_dict)
    start_states = [False] * num_tags
    end_states = [False] * num_tags
    allowed_trainsitions = [[False] * num_tags for _ in range(num_tags)]
    for i, tag_from in tag_dict.items():
        if tag_from.startswith(("B-", "U-")) or tag_from == "O":
            start_states[i] = True
        if tag_from.startswith(("L-", "U-")) or tag_from == "O":
            end_states[i] = True
        prefix_from, entity_from = tag_from[0], tag_from[1:]
        for j, tag_to in tag_dict.items():
            prefix_to, entity_to = tag_to[0], tag_to[1:]
            if is_transition_allowed(
                "BIOUL",
                from_tag=prefix_from,
                from_entity=entity_from,
                to_tag=prefix_to,
                to_entity=entity_to,
            ):
                allowed_trainsitions[i][j] = True

    return start_states, end_states, allowed_trainsitions
