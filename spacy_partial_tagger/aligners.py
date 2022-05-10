from abc import ABC, abstractmethod
from typing import List

from spacy.training.iob_utils import tags_to_entities


class Aligner(ABC):
    @abstractmethod
    def to_subword(self, tags: List[str]) -> List[str]:
        pass

    @abstractmethod
    def from_subword(self, subword_tags: List[str], length: int) -> List[str]:
        pass


class PassThroughAligner(Aligner):
    def to_subword(self, tags: List[str]) -> List[str]:
        return tags

    def from_subword(self, subword_tags: List[str], length: int) -> List[str]:
        return subword_tags


class TransformerAligner(Aligner):
    def __init__(self, mapping: List[List[int]]) -> None:
        self.mapping = mapping

    def to_subword(self, tags: List[str]) -> List[str]:
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
            assert 0 <= i and 0 <= j
            if i == j:
                subword_tags[i] = f"U-{label}"
            else:
                subword_tags[i] = f"B-{label}"
                subword_tags[i + 1 : j] = [f"I-{label}"] * (j - i - 1)
                subword_tags[j] = f"L-{label}"
        return subword_tags

    def from_subword(self, subword_tags: List[str], length: int) -> List[str]:
        tags = ["O"] * length
        # [start, end]
        offsets = tags_to_entities(subword_tags)
        for label, start, end in offsets:
            # [offset_mapping[i][0], offset_mapping[i][1])
            if not self.mapping[start] or not self.mapping[end]:
                continue
            i = self.mapping[start][0]
            j = self.mapping[end][1]
            if i > j:
                continue
            elif i == j:
                tags[i] = f"U-{label}"
            else:
                tags[i] = f"B-{label}"
                tags[i + 1 : j] = [f"I-{label}"] * (j - i - 1)
                tags[j] = f"L-{label}"
        return tags
