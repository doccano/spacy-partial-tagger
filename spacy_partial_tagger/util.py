from typing import List, Tuple, cast

import catalogue
import spacy_alignments as tokenizations
from spacy.tokens import Doc, Span
from spacy.training.iob_utils import biluo_tags_to_offsets
from spacy.util import registry
from transformers import PreTrainedTokenizer

registry.label_indexers = catalogue.create(  # type:ignore
    "spacy", "label_indexers", entry_points=True
)


def get_alignments(
    tokenizer: PreTrainedTokenizer, text: str, input_ids: List[int]
) -> list:
    tokens = tokenizer.word_tokenizer.tokenize(
        text, never_split=tokenizer.all_special_tokens
    )
    _, y2x = tokenizations.get_alignments(text, tokens)
    token2char = {i: (x[0], x[-1] + 1) for i, x in enumerate(y2x)}

    pieces = [
        (i, tokenizer.convert_ids_to_tokens(piece))
        for i, piece in enumerate(input_ids)
        if piece not in {tokenizer.cls_token_id, tokenizer.sep_token_id}
    ]

    mapping: List[List[Tuple[int, int]]] = [[] for _ in range(len(input_ids))]
    now = 0
    for i, token in enumerate(tokens):
        for subword in tokenizer.subword_tokenizer.tokenize(token):
            if pieces[now][1] == subword:
                mapping[pieces[now][0]].append(token2char[i])
                now += 1

    res: List[List[int]] = []
    for m in mapping:
        if not m:
            res.append([])
        else:
            res.append(list(range(m[0][0], m[-1][1])))

    return res


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
