from typing import Any, Callable

from spacy import util
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab
from transformers import AutoTokenizer


class TransformerTokenizer:
    def __init__(self, vocab: Vocab, model_name: str) -> None:
        self.vocab = vocab
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, text: str) -> Doc:
        tokens = self._tokenizer.tokenize(text, add_special_tokens=True)
        return Doc(self.vocab, words=tokens, spaces=[False] * len(tokens))

    def to_bytes(self, *, exclude: tuple = tuple()) -> bytes:
        return b""

    def to_disk(self, path: str, **kwargs: Any) -> None:
        ...

    def from_bytes(
        self, bytes_data: bytes, *, exclude: tuple = tuple()
    ) -> "TransformerTokenizer":
        return self

    def from_disk(
        self, path: str, *, exclude: tuple = tuple()
    ) -> "TransformerTokenizer":
        return self


@util.registry.tokenizers("transformer_tokenizer.v1")
def create_transformer_tokenizer(
    model_name: str,
) -> Callable[[Language], TransformerTokenizer]:
    def create_tokenizer(nlp: Language) -> TransformerTokenizer:
        return TransformerTokenizer(nlp.vocab, model_name)

    return create_tokenizer
