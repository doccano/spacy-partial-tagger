from typing import Any, Callable

from spacy import util
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab


class CharacterTokenizer:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def __call__(self, text: str) -> Doc:
        tokens = list(text)
        return Doc(self.vocab, words=tokens, spaces=[False] * len(tokens))

    def to_bytes(self, *, exclude: tuple = tuple()) -> bytes:
        return b""

    def to_disk(self, path: str, **kwargs: Any) -> None:
        return

    def from_bytes(
        self, bytes_data: bytes, *, exclude: tuple = tuple()
    ) -> "CharacterTokenizer":
        return self

    def from_disk(self, path: str, *, exclude: tuple = tuple()) -> "CharacterTokenizer":
        return self


@util.registry.tokenizers("character_tokenizer.v1")
def create_character_tokenizer() -> Callable[[Language], CharacterTokenizer]:
    def create_tokenizer(nlp: Language) -> CharacterTokenizer:
        return CharacterTokenizer(nlp.vocab)

    return create_tokenizer
