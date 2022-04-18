from typing import List, Tuple

from spacy.training.iob_utils import tags_to_entities


def to_subword_tags(
    tags: List[str], offset_mapping: List[Tuple[int, int]]
) -> List[str]:
    index = [-1] * len(tags)
    for i, (start, end) in enumerate(offset_mapping):
        for j in range(start, end):
            index[j] = i

    offsets = tags_to_entities(tags)
    subword_tags = ["O"] * (sum([1 for x, y in offset_mapping if x != y]) + 2)
    for label, start, end in offsets:
        i = index[start]
        j = index[end]
        if i == j:
            subword_tags[i] = f"U-{label}"
        else:
            subword_tags[i] = f"B-{label}"
            subword_tags[i + 1 : j] = [f"I-{label}"] * (j - i - 1)
            subword_tags[j] = f"L-{label}"
    return subword_tags


def from_subword_tags(
    subword_tags: List[str], offset_mapping: List[Tuple[int, int]], length: int
) -> List[str]:
    tags = ["O"] * length
    offsets = tags_to_entities(subword_tags)
    for label, start, end in offsets:
        i = offset_mapping[start][0]
        j = offset_mapping[end][1] - 1
        if i > j:
            continue
        if i == j:
            tags[i] = f"U-{label}"
        else:
            tags[i] = f"B-{label}"
            tags[i + 1 : j] = [f"I-{label}"] * (j - i - 1)
            tags[j] = f"L-{label}"
    return tags
