from typing import List

import catalogue
from spacy.util import registry
from tokenizations import get_alignments as get_alignments_original

registry.label_indexers = catalogue.create(  # type:ignore
    "spacy", "label_indexers", entry_points=True
)


def get_alignments(source: List[str], target: List[str]) -> List[List[int]]:
    _, y2x = get_alignments_original(source, target)
    indices = iter(range(len(source)))
    for y in y2x:
        if not y:
            # TODO: Fix me, maybe this doesn't work properly when [UNK] corresponds
            # more than 1 letter.
            for i in indices:
                if source[i] != " ":
                    y.append(i)
                    break
        else:
            for j in y:
                if j in indices:
                    continue
    return y2x
