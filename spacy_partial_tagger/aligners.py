from abc import ABC, abstractmethod
from itertools import groupby
from typing import List, Tuple

from spacy.training.iob_utils import tags_to_entities


class Aligner(ABC):
    """Base class for all aligners."""

    @abstractmethod
    def to_subword(self, tags: List[str]) -> List[str]:
        pass

    @abstractmethod
    def from_subword(self, subword_tags: List[str]) -> List[str]:
        pass


class PassThroughAligner(Aligner):
    """Aligner for compatibility."""

    def to_subword(self, tags: List[str]) -> List[str]:
        return tags

    def from_subword(self, subword_tags: List[str]) -> List[str]:
        return subword_tags


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
                f"Multiple labels ({labels_char}) are assigned"
                + f"for the sub-word at {i_target}."
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


class CharHopAliginer(Aligner):
    def __init__(
        self,
        char_offsets_token: List[List[int]],
        char_offsets_subword: List[List[int]],
        char_length: int,
        token_length: int,
        subword_length: int,
    ) -> None:

        self.char_offsets_token = char_offsets_token
        self.char_offsets_subword = char_offsets_subword
        self.char_length = char_length
        self.token_length = token_length
        self.subword_length = subword_length

    def to_subword(self, tags: List[str]) -> List[str]:
        return convert_tags(
            tags,
            self.char_offsets_token,
            self.char_offsets_subword,
            self.char_length,
            self.subword_length,
            False,
        )

    def from_subword(self, subword_tags: List[str]) -> List[str]:
        # TODO: check if boundaries are fine.
        return convert_tags(
            subword_tags,
            self.char_offsets_subword,
            self.char_offsets_token,
            self.char_length,
            self.token_length,
            True,
        )


class TransformerAligner(Aligner):
    """
    Aligner for Transformers

    Args:
        mapping: A list of list of integers representing offset mapping.

    """

    def __init__(self, mapping: List[List[int]], length: int = 0) -> None:
        self.mapping = mapping
        self.length = length or (
            max(index for indices in mapping for index in indices) + 1
        )

    def to_subword(self, tags: List[str]) -> List[str]:
        """Converts token-based tags to sub-word-based tags.

        Args:
            tags: A list of string representing tag sequence.
        """
        index = [-1] * len(tags)
        for i, start_end in enumerate(self.mapping):
            if not start_end:
                continue
            # [start, end)
            for j in start_end:
                index[j] = i

        offsets = tags_to_entities(tags)
        subword_tags = ["O"] * len(self.mapping)
        for label, start, end in offsets:
            # [start, end]
            i = index[start]
            j = index[end]
            assert 0 <= i and 0 <= j, (self.mapping, tags, offsets)
            if i == j:
                subword_tags[i] = f"U-{label}"
            else:
                subword_tags[i] = f"B-{label}"
                subword_tags[i + 1 : j] = [f"I-{label}"] * (j - i - 1)
                subword_tags[j] = f"L-{label}"
        return subword_tags

    def from_subword(self, subword_tags: List[str]) -> List[str]:
        """Converts sub-word-based tags to token-based tags.

        Args:
            subword_tags: A list of string representing tag sequence.
        """
        tags = ["O"] * self.length
        # [start, end]
        offsets = tags_to_entities(subword_tags[: len(self.mapping)])
        for label, start, end in offsets:
            if not self.mapping[start] or not self.mapping[end]:
                continue
            # [mapping[i][0], mapping[i][-1]]
            i = self.mapping[start][0]
            j = self.mapping[end][-1]
            if i > j:
                continue
            elif i >= self.length or j >= self.length:
                break
            elif i == j:
                tags[i] = f"U-{label}"
            else:
                tags[i] = f"B-{label}"
                tags[i + 1 : j] = [f"I-{label}"] * (j - i - 1)
                tags[j] = f"L-{label}"
        return tags
