import json
import os
from argparse import ArgumentParser
from typing import List, Tuple

import spacy
from spacy.tokens import DocBin
from spacy.training.iob_utils import biluo_tags_to_spans

from .tokenizer import TransformerTokenizer


def converter(
    tokens: List[str], text: str, subwords: List[str], annotations: List[dict]
) -> List[Tuple[int, int, str]]:
    """Converts annotations from token-based to subword-based.

    Args:
        tokens: A list of tokens.
        text: A text.
        subwords: A list of subowrds.
        annotations: A list of dictionaries represent a named entity  annotation.
        Each dictionary should have start/end/type keys. Start and end represent
        start position and end position respectively. Type represents a label of
        a named entity.

    Returns:
        A list of subword-based annotations.
    """

    # TODO: Split into some functions
    char_tokens = list(text)

    char_index_to_subword_index: list = [-1] * len(char_tokens)
    now = 0
    for i, subword in enumerate(subwords):
        for char in subword:
            now = char_tokens.index(char, now)
            char_index_to_subword_index[now] = i
            now += 1
    token_index_to_subword_index: list = [[] for _ in range(len(tokens))]
    now = 0
    for i, token in enumerate(tokens):
        for char in token:
            now = char_tokens.index(char, now)
            if (
                not token_index_to_subword_index[i]
                or token_index_to_subword_index[i][-1]
                != char_index_to_subword_index[now]
            ):
                token_index_to_subword_index[i].append(char_index_to_subword_index[now])
            now += 1
    subword_annotations = [
        # [start, end)
        (
            token_index_to_subword_index[annotation["start"]][0],
            token_index_to_subword_index[annotation["end"] - 1][-1] + 1,
            annotation["type"],
        )
        for annotation in annotations
    ]
    return subword_annotations


def main(input_path: str, output_path: str, lang: str, model_name: str) -> None:

    print("Initializing...")
    nlp = spacy.blank(lang)
    tokenizer = TransformerTokenizer(nlp.vocab, model_name)
    db = DocBin()

    print("Processing...")
    with open(input_path) as f:
        for data in map(json.loads, f):
            subwords = tokenizer(" ".join(data["tokens"]))
            annotations = converter(
                data["tokens"],
                subwords.text,
                [subword.text for subword in subwords],
                data["gold_annotations"],
            )
            tags = ["O"] * len(subwords)
            for start, end, entity in annotations:
                if any(tag != "O" for tag in tags[start:end]):
                    continue
                if start + 1 == end:
                    tags[start:end] = [f"U-{entity}"]
                    continue
                tags[start:end] = (
                    [f"B-{entity}"]
                    + [f"I-{entity}"] * (end - start - 2)
                    + [f"L-{entity}"]
                )
            subwords.ents = biluo_tags_to_spans(subwords, tags)  # type:ignore
            db.add(subwords)

    db.to_disk(output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", help="Language")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilroberta-base",
        help="Transformers model name",
    )
    parser.add_argument("--input", type=str, required=True, help="An input file path")
    parser.add_argument("--output", type=str, default=None, help="An output file path")

    args = parser.parse_args()

    if args.output is None:
        basename, _ = os.path.splitext(args.input)
        output = f"{basename}.spacy"
    else:
        output = args.output

    main(args.input, output, args.lang, args.model_name)
