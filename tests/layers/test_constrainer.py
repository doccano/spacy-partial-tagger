import numpy as np
import pytest
from thinc.backends.numpy_ops import NumpyOps

from spacy_partial_tagger.layers.constrainer import (
    get_subword_end_state,
    get_subword_inside_state,
    get_subword_outside_state,
    get_subword_start_state,
    get_subword_unit_state,
    get_transition_constraints,
)


@pytest.fixture
def tag_dict() -> dict:
    return {
        i: token
        for i, token in enumerate(
            [
                "O",
                "B-PER",
                "I-PER",
                "L-PER",
                "U-PER",
                "B-LOC",
                "I-LOC",
                "L-LOC",
                "U-LOC,",
            ]
        )
    }


@pytest.fixture
def ops() -> NumpyOps:
    return NumpyOps()


def test_get_transition_constraints(tag_dict: dict, ops: NumpyOps) -> None:
    assert np.array_equal(
        get_transition_constraints(ops, tag_dict),
        np.array(
            [
                [1, 1, 0, 0, 1, 1, 0, 0, 1],  # O
                [0, 0, 1, 1, 0, 0, 0, 0, 0],  # B-PER
                [0, 0, 1, 1, 0, 0, 0, 0, 0],  # I-PER
                [1, 1, 0, 0, 1, 1, 0, 0, 1],  # L-PER
                [1, 1, 0, 0, 1, 1, 0, 0, 1],  # U-PER
                [0, 0, 0, 0, 0, 0, 1, 1, 0],  # B-LOC
                [0, 0, 0, 0, 0, 0, 1, 1, 0],  # I-LOC
                [1, 1, 0, 0, 1, 1, 0, 0, 1],  # L-LOC
                [1, 1, 0, 0, 1, 1, 0, 0, 1],  # U-LOC
            ]
        ),
    )


def test_get_subword_outside_state(tag_dict: dict, ops: NumpyOps) -> None:
    assert np.array_equal(
        get_subword_outside_state(ops, tag_dict), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    )


def test_get_subword_start_state(tag_dict: dict, ops: NumpyOps) -> None:
    assert np.array_equal(
        get_subword_start_state(ops, tag_dict), np.array([1, 1, 1, 0, 0, 1, 1, 0, 0])
    )


def test_get_subword_inside_state(tag_dict: dict, ops: NumpyOps) -> None:
    assert np.array_equal(
        get_subword_inside_state(ops, tag_dict), np.array([1, 0, 1, 0, 0, 0, 1, 0, 0])
    )


def test_get_subword_end_state(tag_dict: dict, ops: NumpyOps) -> None:
    assert np.array_equal(
        get_subword_end_state(ops, tag_dict), np.array([1, 0, 1, 1, 0, 0, 1, 1, 0])
    )


def test_get_subword_unit_state(tag_dict: dict, ops: NumpyOps) -> None:
    assert np.array_equal(
        get_subword_unit_state(ops, tag_dict), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    )
