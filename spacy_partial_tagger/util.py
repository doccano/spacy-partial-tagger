from typing import List, Tuple

import catalogue
import spacy_alignments as tokenizations
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
        offset = 0
        for subword in tokenizer.subword_tokenizer.tokenize(token):
            if pieces[now][1] == subword:
                cleaned_subword = subword.replace("##", "")
                start, end = token2char[i]
                mapping[pieces[now][0]].append(
                    (start + offset, min(end, start + offset + len(cleaned_subword)))
                )
                now += 1
                offset += len(cleaned_subword)

    res: List[List[int]] = []
    for m in mapping:
        if not m:
            res.append([])
        else:
            res.append(list(range(m[0][0], m[-1][1])))
    return res
