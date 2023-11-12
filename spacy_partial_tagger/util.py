from typing import Dict, List, Tuple, cast

import spacy_alignments as tokenizations
import torch
from partial_tagger.crf import functional as F
from partial_tagger.crf.nn import CRF
from partial_tagger.encoders.base import BaseEncoder
from partial_tagger.encoders.transformer import TransformerModelEncoderFactory
from sequence_label import LabelSet
from torch import nn
from transformers import PreTrainedTokenizer


class SequenceTagger(nn.Module):
    def __init__(
        self,
        encoder: BaseEncoder,
        padding_index: int,
        start_states: Tuple[bool, ...],
        end_states: Tuple[bool, ...],
        transitions: Tuple[Tuple[bool, ...], ...],
    ):
        super().__init__()

        self.encoder = encoder
        self.crf = CRF(encoder.get_hidden_size())
        self.start_constraints = nn.Parameter(
            torch.tensor(start_states), requires_grad=False
        )
        self.end_constraints = nn.Parameter(
            torch.tensor(end_states), requires_grad=False
        )
        self.transition_constraints = nn.Parameter(
            torch.tensor(transitions), requires_grad=False
        )
        self.padding_index = padding_index

    def __constrain(
        self, log_potentials: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return F.constrain_log_potentials(
            log_potentials,
            mask,
            self.start_constraints,
            self.end_constraints,
            self.transition_constraints,
        )

    def forward(
        self, inputs: Dict[str, torch.Tensor], mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_potentials = self.crf(self.encoder(inputs), mask)

        contrained = self.__constrain(log_potentials, mask)

        contrained.requires_grad_()

        with torch.enable_grad():
            _, tag_indices = F.decode(contrained)

        return log_potentials, tag_indices * mask + self.padding_index * (~mask)

    def predict(
        self, inputs: Dict[str, torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        return cast(torch.Tensor, self(inputs, mask)[1])


def create_tagger(
    model_name: str, label_set: LabelSet, padding_index: int
) -> SequenceTagger:
    return SequenceTagger(
        TransformerModelEncoderFactory(model_name).create(label_set),
        padding_index,
        label_set.start_states,
        label_set.end_states,
        label_set.transitions,
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
