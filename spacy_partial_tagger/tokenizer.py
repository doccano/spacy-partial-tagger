from typing import Optional, Tuple

from partial_tagger.data import Alignment, Span
from partial_tagger.data.batch.text import (
    BaseTokenizer,
    TextBatch,
    TransformerTokenizer,
)
from transformers import AutoTokenizer
from transformers.models.bert_japanese import (
    BertJapaneseTokenizer as _BertJapaneseTokenizer,
)

from .util import get_alignments


class BertJapaneseTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer: _BertJapaneseTokenizer,
        tokenizer_args: Optional[dict] = None,
    ):
        self.__tokenizer = tokenizer

        self.__tokenizer_args = tokenizer_args or {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        self.__tokenizer_args["return_offsets_mapping"] = True

    def __call__(self, texts: Tuple[str]) -> TextBatch:
        batch_encoding = self.__tokenizer(texts, **self.__tokenizer_args)

        pad_token_id = self.__tokenizer.pad_token_id
        mask = batch_encoding.input_ids != pad_token_id
        tokenized_text_lengths = mask.sum(dim=1)

        alignments = []
        for _tokenized_text_length, input_ids, text in zip(
            tokenized_text_lengths, batch_encoding.input_ids, texts
        ):
            char_spans = tuple(
                Span(span[0], len(span)) if span else None
                for span in get_alignments(self.__tokenizer, text, input_ids.tolist())
            )
            token_indices = [-1] * len(text)
            for token_index, char_span in enumerate(char_spans):
                if char_span is None:
                    continue
                start = char_span.start
                end = char_span.start + char_span.length
                token_indices[start:end] = [token_index] * char_span.length

            alignments.append(Alignment(text, char_spans, tuple(token_indices)))

        return TextBatch(
            tagger_inputs=batch_encoding, mask=mask, alignments=tuple(alignments)
        )


def get_tokenizer(
    transformer_model_name: str, tokenizer_args: Optional[dict] = None
) -> BaseTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
    if isinstance(tokenizer, _BertJapaneseTokenizer):
        return BertJapaneseTokenizer(tokenizer, tokenizer_args)
    else:
        return TransformerTokenizer(tokenizer, tokenizer_args)
