from typing import List, Tuple

import spacy_alignments as tokenizations
from partial_tagger.data import LabelSet
from partial_tagger.decoders.viterbi import Constrainer, ViterbiDecoder
from partial_tagger.encoders.transformer import TransformerModelEncoderFactory
from partial_tagger.tagger import SequenceTagger
from transformers import PreTrainedTokenizer


def create_tagger(
    model_name: str, label_set: LabelSet, padding_index: int
) -> SequenceTagger:
    return SequenceTagger(
        TransformerModelEncoderFactory(model_name).create(label_set),
        ViterbiDecoder(
            padding_index,
            Constrainer(
                label_set.get_start_states(),
                label_set.get_end_states(),
                label_set.get_transitions(),
            ),
        ),
    )


def get_alignments(
    tokenizer: PreTrainedTokenizer, text: str, input_ids: List[int]
) -> list:
    tokens = tokenizer.word_tokenizer.tokenize(
        text, never_split=tokenizer.all_special_tokens
    )
    _, y2x = tokenizations.get_alignments(list(text), tokens)
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
