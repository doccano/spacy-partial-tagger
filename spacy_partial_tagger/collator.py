from typing import List, Optional, Tuple

from partial_tagger.data.collators import BaseCollator, Batch, TransformerCollator
from sequence_label.core import LabelAlignment, Span
from transformers import AutoTokenizer
from transformers.models.bert_japanese import BertJapaneseTokenizer

from .util import get_alignments


class BertJapaneseCollator(BaseCollator):
    def __init__(
        self,
        tokenizer: BertJapaneseTokenizer,
        tokenizer_args: Optional[dict] = None,
    ):
        self.__tokenizer = tokenizer

        self.__tokenizer_args = tokenizer_args or {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        self.__tokenizer_args["return_offsets_mapping"] = True

    def __call__(
        self, texts: Tuple[str, ...]
    ) -> Tuple[Batch, Tuple[LabelAlignment, ...]]:
        batch_encoding = self.__tokenizer(texts, **self.__tokenizer_args)

        pad_token_id = self.__tokenizer.pad_token_id
        mask = batch_encoding.input_ids != pad_token_id

        alignments = []
        for input_ids, text in zip(batch_encoding.input_ids, texts):
            char_spans = tuple(
                Span(start=span[0], length=len(span)) if span else None
                for span in get_alignments(self.__tokenizer, text, input_ids.tolist())
            )
            token_spans: List[Optional[Span]] = [None] * len(text)
            for token_index, char_span in enumerate(char_spans):
                if char_span is None:
                    continue
                start = char_span.start
                end = char_span.start + char_span.length
                for i in range(start, end):
                    token_spans[i] = Span(start=token_index, length=1)

            alignments.append(LabelAlignment(char_spans, tuple(token_spans)))

        return Batch(tagger_inputs=batch_encoding, mask=mask), tuple(alignments)


def get_collator(
    transformer_model_name: str, tokenizer_args: Optional[dict] = None
) -> BaseCollator:
    tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
    if isinstance(tokenizer, BertJapaneseTokenizer):
        return BertJapaneseCollator(tokenizer, tokenizer_args)
    else:
        return TransformerCollator(tokenizer, tokenizer_args)
