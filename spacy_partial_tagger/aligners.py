from abc import ABCMeta, abstractmethod
from itertools import groupby
from typing import List, Tuple

from spacy.training.iob_utils import tags_to_entities


class Aligner(metaclass=ABCMeta):
    """Base class for all aligners."""

    @abstractmethod
    def to_subword(self, batch_tags: List[List[str]]) -> List[List[str]]:
        pass

    @abstractmethod
    def from_subword(self, batch_subword_tags: List[List[str]]) -> List[List[str]]:
        pass


def convert_tags(
    tags_source: List[str],
    char_offsets_source: List[Tuple[int, int]],
    char_offsets_target: List[Tuple[int, int]],
    char_length: int,
    target_length: int,
    strict: bool = False,
) -> List[str]:
    labels_char = ["O"] * char_length
    for label, start_source, end_source in tags_to_entities(tags_source):
        start_char = char_offsets_source[start_source][0]
        end_char = char_offsets_source[end_source][-1]
        labels_char[start_char:end_char] = [label] * (end_char - start_char)

    labels_target = ["O"] * target_length
    for i_target, (start_char, end_char) in enumerate(char_offsets_target):
        if start_char == end_char:
            continue
        labels_unique = set(labels_char[start_char:end_char])
        if len(labels_unique) == 1:
            labels_target[i_target] = labels_unique.pop()
        elif not strict:
            # add warnings
            labels_target[i_target] = labels_char[start_char]
        else:
            raise ValueError(
                f"Multiple labels ({labels_char[start_char:end_char]}) are assigned"
                + f" for the sub-word at {i_target}."
            )

    # maybe Tags(labels, BILUO)
    tags_target = []
    for label, group in groupby(labels_target):
        group_size = len(list(group))
        if label == "O":
            tags_target.extend([label] * group_size)
        elif group_size == 1:
            tags_target.append(f"U-{label}")
        else:
            tags_target.append(f"B-{label}")
            tags_target.extend([f"I-{label}"] * (group_size - 2))
            tags_target.append(f"L-{label}")

    assert len(tags_target) == target_length

    return tags_target


class TransformerAligner(Aligner):
    """
    Aligner for Transformers

    Args:
        mapping: A list of list of integers representing offset mapping.

    """

    def __init__(
        self,
        batch_char_offsets_token: List[List[Tuple[int, int]]],
        batch_char_offsets_subword: List[List[Tuple[int, int]]],
        char_lengths: List[int],
        token_lengths: List[int],
        subword_lengths: List[int],
    ) -> None:

        self.batch_char_offsets_token = batch_char_offsets_token
        self.batch_char_offsets_subword = batch_char_offsets_subword
        self.char_lengths = char_lengths
        self.token_lengths = token_lengths
        self.subword_lengths = subword_lengths

    def to_subword(self, batch_tags: List[List[str]]) -> List[List[str]]:
        """Converts token-based tags to sub-word-based tags.

        Args:
            batch_tags: A list of string representing tag sequence.
        """
        batch_subword_tags = []
        for (
            tags,
            char_offsets_token,
            char_offsets_subword,
            char_length,
            subword_length,
        ) in zip(
            batch_tags,
            self.batch_char_offsets_token,
            self.batch_char_offsets_subword,
            self.char_lengths,
            self.subword_lengths,
        ):
            batch_subword_tags.append(
                convert_tags(
                    tags,
                    char_offsets_token,
                    char_offsets_subword,
                    char_length,
                    subword_length,
                    False,
                )
            )
        return batch_subword_tags

    def from_subword(self, batch_subword_tags: List[List[str]]) -> List[List[str]]:
        """Converts sub-word-based tags to token-based tags.

        Args:
            subword_tags: A list of string representing tag sequence.
        """
        batch_tags = []
        for (
            subword_tags,
            char_offsets_subword,
            char_offsets_token,
            char_length,
            token_length,
        ) in zip(
            batch_subword_tags,
            self.batch_char_offsets_subword,
            self.batch_char_offsets_token,
            self.char_lengths,
            self.token_lengths,
        ):
            batch_tags.append(
                convert_tags(
                    subword_tags,
                    char_offsets_subword,
                    char_offsets_token,
                    char_length,
                    token_length,
                    False,
                )
            )
        return batch_tags
