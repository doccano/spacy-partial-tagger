from abc import ABC, abstractmethod
from typing import Dict, List

from .util import registry


class LabelIndexer(ABC):
    """
    Base class for indexing NER tags.
    """

    @abstractmethod
    def __call__(
        self, batch_tags: List[List[str]], tag_to_id: Dict[str, int]
    ) -> List[List[int]]:
        pass


class TransformerLabelIndexer(LabelIndexer):
    """
    Indexer for Transformers model.

    Args:
        padding_index: an integer representing a padding index.
        unknown_index: an integer representing that a label is unavailable.
    """

    def __init__(self, padding_index: int, unknown_index: int) -> None:
        self.padding_index = padding_index
        self.unknown_index = unknown_index

    def __call__(
        self, batch_tags: List[List[str]], tag_to_id: Dict[str, int]
    ) -> List[List[int]]:
        batch_tag_indices = []
        for tags in batch_tags:
            # Indexing
            tag_indices = [
                tag_to_id[tag] if tag != "O" else self.unknown_index for tag in tags
            ]
            # always assign O tags to start/end token.
            tag_indices[0] = tag_indices[-1] = tag_to_id["O"]
            batch_tag_indices.append(tag_indices)

        # Padding
        max_length = max(map(len, batch_tag_indices))
        return [
            tag_indices + [self.padding_index] * (max_length - len(tag_indices))
            for tag_indices in batch_tag_indices
        ]


@registry.label_indexers(  # type:ignore
    "spacy-partial-tagger.TransformerLabelIndexer.v1"
)
def configure_transformer_label_indexer(
    padding_index: int, unknown_index: int
) -> TransformerLabelIndexer:
    return TransformerLabelIndexer(padding_index, unknown_index)
