from abc import ABC, abstractmethod
from typing import Dict, List

from spacy.tokens import Doc
from spacy.training.iob_utils import iob_to_biluo

from .util import registry


class LabelIndexer(ABC):
    @abstractmethod
    def __call__(self, docs: List[Doc], tag_to_id: Dict[str, int]) -> List[List[int]]:
        pass


class RoBERTaLabelIndexer(LabelIndexer):
    def __init__(self, padding_index: int, unknown_index: int) -> None:
        self.padding_index = padding_index
        self.unknown_index = unknown_index

    def __call__(self, docs: List[Doc], tag_to_id: Dict[str, int]) -> List[List[int]]:
        batch_tag_indices = []
        for doc in docs:
            tags = iob_to_biluo(
                [
                    f"{token.ent_iob_}-{token.ent_type_}"
                    if token.ent_iob_ != "O"
                    else "O"
                    for token in doc
                ]
            )
            # Indexing
            tag_indices = [
                tag_to_id[tag] if tag != "O" else self.unknown_index for tag in tags
            ]
            tag_indices[0] = tag_indices[-1] = tag_to_id["O"]
            batch_tag_indices.append(tag_indices)

        # Padding
        max_length = max(map(len, batch_tag_indices))
        return [
            tag_indices + [self.padding_index] * (max_length - len(tag_indices))
            for tag_indices in batch_tag_indices
        ]


@registry.label_indexers("spacy-partial-tagger.RoBERTaLabelIndexer.v1")  # type:ignore
def configure_roberta_label_indexer(
    padding_index: int, unknown_index: int
) -> RoBERTaLabelIndexer:
    return RoBERTaLabelIndexer(padding_index, unknown_index)
