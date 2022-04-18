import json
import os
from argparse import ArgumentParser
from typing import List

import spacy
from spacy.tokens import Doc, DocBin
from spacy.training.iob_utils import biluo_tags_to_spans


def converter(tokens: List[str], annotations: List[dict]) -> tuple:
    char_tokens = list(" ".join(tokens))
    token_index_to_char_index: list = [[] for _ in range(len(tokens))]
    now = 0
    for i, token in enumerate(tokens):
        for char in token:
            now = char_tokens.index(char, now)
            token_index_to_char_index[i].append(now)
    char_annotations = [
        # [start, end)
        (
            token_index_to_char_index[annotation["start"]][0],
            token_index_to_char_index[annotation["end"] - 1][-1] + 1,
            annotation["type"],
        )
        for annotation in annotations
    ]
    return char_tokens, char_annotations


def main(input_path: str, output_path: str, lang: str) -> None:

    nlp = spacy.blank(lang)
    db = DocBin()

    with open(input_path) as f:
        for data in map(json.loads, f):
            tokens, annotations = converter(data["tokens"], data["gold_annotations"])
            tags = ["O"] * len(tokens)
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
            doc = Doc(
                nlp.vocab,
                words=tokens,
                spaces=[False] * len(tokens),
            )
            doc.ents = biluo_tags_to_spans(doc, tags)  # type:ignore
            db.add(doc)

    db.to_disk(output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", help="Language")
    parser.add_argument("--input", type=str, required=True, help="An input file path")
    parser.add_argument("--output", type=str, default=None, help="An output file path")

    args = parser.parse_args()

    if args.output is None:
        basename, _ = os.path.splitext(args.input)
        output = f"{basename}.spacy"
    else:
        output = args.output

    main(args.input, output, args.lang)
