from typing import Any, Dict

from spacy.tokens import Doc
from thinc.api import Ops
from thinc.types import Floats4d, Ints1d, Ints2d, Ints3d
from thinc.util import get_array_module


def get_token_mapping(doc: Doc, mapping: list) -> list:
    token = []
    char = [list(range(token.idx, token.idx + len(token.text))) for token in doc]
    index = 0
    indices = []
    now = []
    for i, x in enumerate(mapping):
        if not x:
            token.append([i])
            continue
        indices.append(i)
        now += x
        if now == char[index]:
            index += 1
            token.append(indices[:])
            indices.clear()
            now.clear()
    return token


def get_transition_constraints(ops: Ops, tag_dict: dict) -> Ints2d:
    mask = ops.alloc2i(len(tag_dict), len(tag_dict), zeros=True)
    for i, prev in tag_dict.items():
        for j, now in tag_dict.items():
            if prev.startswith(("U-", "L-", "O")) and now.startswith(("U-", "B-", "O")):
                mask[i][j] = 1
            elif (
                prev.startswith(("B-", "I-"))
                and now.startswith(("I-", "L-"))
                and prev[2:] == now[2:]
            ):
                mask[i][j] = 1
    return mask


def get_subword_outside_state(ops: Ops, tag_dict: dict) -> Ints1d:
    state = ops.alloc1i(len(tag_dict), zeros=True)
    for i, tag in tag_dict.items():
        if tag == "O":
            state[i] = 1
    return state


def get_subword_start_state(ops: Ops, tag_dict: dict) -> Ints1d:
    state = ops.alloc1i(len(tag_dict), zeros=True)
    for i, tag in tag_dict.items():
        if tag.startswith("B-") or tag == "O":
            state[i] = 1
    return state


def get_subword_inside_state(ops: Ops, tag_dict: dict) -> Ints1d:
    state = ops.alloc1i(len(tag_dict), zeros=True)
    for i, tag in tag_dict.items():
        if tag.startswith("I-") or tag == "O":
            state[i] = 1
    return state


def get_subword_end_state(ops: Ops, tag_dict: dict) -> Ints1d:
    state = ops.alloc1i(len(tag_dict), zeros=True)
    for i, tag in tag_dict.items():
        if tag.startswith("L-") or tag == "O":
            state[i] = 1
    return state


def get_subword_unit_state(ops: Ops, tag_dict: dict) -> Ints1d:
    state = ops.alloc1i(len(tag_dict), zeros=True)
    for i, tag in tag_dict.items():
        if tag.startswith("U-") or tag == "O":
            state[i] = 1
    return state


class Constrainer:
    def __init__(self, tag_dict: dict, mappings: list):
        self.tag_dict = tag_dict
        self.mappings = mappings

    def __call__(self, ops: Ops, log_potentials: Floats4d) -> Floats4d:
        transition = get_transition_constraints(ops, self.tag_dict)
        xp = get_array_module(log_potentials)
        for i, mapping in enumerate(self.mappings):
            constraint = self._mapping_to_constraint(ops, xp, mapping)
            constraint *= transition[None]  # type:ignore
            log_potentials[i, : constraint.shape[0]] *= constraint
        return log_potentials

    def _mapping_to_constraint(self, ops: Ops, xp: Any, mapping: list) -> Ints3d:
        states = [get_subword_outside_state(ops, self.tag_dict)] * 2
        for indices in mapping[1:-1]:  # ignore start and end tokens
            length = len(indices)
            if length == 0:
                continue
            elif length == 1:
                states.append(get_subword_unit_state(ops, self.tag_dict))
            else:
                states.append(get_subword_start_state(ops, self.tag_dict))
                for _ in range(length - 2):
                    states.append(get_subword_inside_state(ops, self.tag_dict))
                states.append(get_subword_end_state(ops, self.tag_dict))
        states.append(get_subword_outside_state(ops, self.tag_dict))

        prev = xp.stack(states[:-1])
        now = xp.stack(states[1:])
        constraint = prev[..., None] * now[:, None]

        return constraint


class ConstrainerFactory:
    def __init__(self, tag_dict: Dict[int, str]):
        self.tag_dict = tag_dict

    def get_constrainer(self, mappings: list) -> Constrainer:
        return Constrainer(self.tag_dict, mappings)
