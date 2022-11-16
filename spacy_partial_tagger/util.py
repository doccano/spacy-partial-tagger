from typing import List, Tuple, cast

import catalogue
import spacy_alignments as tokenizations
from spacy.tokens import Doc, Span
from spacy.training.iob_utils import biluo_tags_to_offsets
from spacy.util import registry

registry.label_indexers = catalogue.create(  # type:ignore
    "spacy", "label_indexers", entry_points=True
)


def get_alignments(source: List[str], target: List[str]) -> List[List[int]]:
    _, y2x = tokenizations.get_alignments(source, target)
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


def make_char_based_doc(doc: Doc, tags: List[str]) -> Doc:
    """Converts token-based doc to character-based doc.

    Args:
        doc: A token-based doc.
        tags: A list of string representing a NER tag in BILUO format.

    Returns:
        A character-based doc
    """
    text = doc.text

    chars = list(text)

    char_doc = Doc(doc.vocab, words=chars, spaces=[False] * len(chars))

    ents = []
    for start, end, label in biluo_tags_to_offsets(doc, tags):
        ents.append(Span(char_doc, start, end, label=label))

    char_doc.ents = cast(Tuple[Span], tuple(ents))

    return char_doc
