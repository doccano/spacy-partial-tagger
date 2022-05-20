from collections.abc import Iterable

from spacy.training.batchers import BatcherT, Sizing
from spacy.util import registry


@registry.batchers("spacy-partial-tagger.batch_by_sequence.v1")
def configure_minibatch(size: Sizing, max_length: int) -> BatcherT:
    def batcher(items: Iterable) -> Iterable:
        batch = []
        for item in filter(lambda item: len(item) <= max_length, items):
            batch.append(item)
            if len(batch) == size:
                yield batch
                batch.clear()
        if batch:
            yield batch

    return batcher
