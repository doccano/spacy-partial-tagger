from typing import Any, Callable, Dict, Optional, Tuple, cast

import torch
from partial_tagger.decoders.viterbi import ConstrainedViterbiDecoder
from spacy.util import registry
from thinc.api import ArgsKwargs, Model, torch2xp, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import Floats4d, Ints1d, Ints2d

from .util import get_mask


@registry.architectures.register("spacy-partial-tagger.ConstrainedViterbiDecoder.v1")
def build_constrained_viterbi_decoder_v1(
    padding_index: int = -1,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None,
) -> Model[Tuple[Floats4d, Ints1d], Ints2d]:
    return Model(
        name="constrained_viterbi_decoder",
        forward=forward,
        init=init,
        attrs={
            "padding_index": padding_index,
            "mixed_precision": mixed_precision,
            "grad_scaler": grad_scaler,
        },
    )


def forward(
    model: Model[Tuple[Floats4d, Ints1d], Ints2d],
    X: Tuple[Floats4d, Ints1d],
    is_train: bool,
) -> Tuple[Ints2d, Callable]:
    return model.layers[0](X, is_train)


def init(
    model: Model[Tuple[Floats4d, Ints1d], Ints2d], X: Any = None, Y: Any = None
) -> None:
    if model.layers:
        return

    if Y is None:
        Y = {0: "O"}

    PyTorchWrapper = registry.get("layers", "PyTorchWrapper.v2")

    padding_index = model.attrs["padding_index"]
    mixed_precision = model.attrs["mixed_precision"]
    grad_scaler = model.attrs["grad_scaler"]

    decoder = PyTorchWrapper(
        ConstrainedViterbiDecoder(*get_constraints(Y), padding_index=padding_index),
        mixed_precision=mixed_precision,
        convert_inputs=convert_inputs,
        convert_outputs=convert_outputs,
        grad_scaler=grad_scaler,
    )

    model._layers = [decoder]


def convert_inputs(
    model: Model[Tuple[Floats4d, Ints1d], Ints2d],
    X_lengths: Tuple[Floats4d, Ints1d],
    is_train: bool,
) -> Tuple[ArgsKwargs, Callable]:
    X, L = X_lengths

    Xt = xp2torch(X, requires_grad=True)
    Lt = xp2torch(L, requires_grad=False)
    mask = get_mask(Lt, Xt.size(1), Xt.device)
    output = ArgsKwargs(args=(Xt, mask), kwargs={})
    return output, lambda d_inputs: []


def convert_outputs(
    model: Model[Tuple[Floats4d, Ints1d], Ints2d],
    inputs_outputs: Tuple[Tuple[Floats4d, Ints1d], torch.Tensor],
    is_train: bool,
) -> Tuple[Ints2d, Callable]:
    _, Y_t = inputs_outputs
    Y = cast(Ints2d, torch2xp(Y_t))
    return Y, lambda dY: []


# Copied from AllenNLP
def is_transition_allowed(
    from_tag: str, from_entity: str, to_tag: str, to_entity: str
) -> bool:
    return any(
        [
            # O can transition to O, B-* or U-*
            # L-x can transition to O, B-*, or U-*
            # U-x can transition to O, B-*, or U-*
            from_tag in ("O", "L", "U") and to_tag in ("O", "B", "U"),
            # B-x can only transition to I-x or L-x
            # I-x can only transition to I-x or L-x
            from_tag in ("B", "I")
            and to_tag in ("I", "L")
            and from_entity == to_entity,
        ]
    )


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
                from_tag=prefix_from,
                from_entity=entity_from,
                to_tag=prefix_to,
                to_entity=entity_to,
            ):
                allowed_trainsitions[i][j] = True

    return start_states, end_states, allowed_trainsitions
