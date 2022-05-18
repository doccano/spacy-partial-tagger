import os
from argparse import ArgumentParser
from typing import Iterable, List, Tuple

import spacy
from conllu import parse_incr
from mojimoji import han_to_zen
from pyknp import Juman
from spacy.tokens import Doc, DocBin
from spacy.training.iob_utils import biluo_tags_to_spans, tags_to_entities
from tokenizations import get_alignments

from .convert import converter

jumanpp = Juman()


def tokenize(text: str) -> List[str]:
    return [mrph.midasi for mrph in jumanpp.analysis(text).mrph_list()]


def load_from_conllu(
    path: str,
) -> Iterable[Tuple[List[str], List[dict]]]:
    with open(path) as f:
        for data in parse_incr(f):
            tags = ["O"] * len(data)
            text = ""
            tokens = []
            for i, x in enumerate(data):
                if "NE" in x["misc"]:
                    tags[i] = x["misc"]["NE"]
                tokens.append(han_to_zen(x["form"]))
                text += x["form"]
                if x["misc"]["SpaceAfter"] == "Yes":
                    # Use full-width space
                    text += "　"
            text = han_to_zen(text)
            tokens_juman = tokenize(text)
            # x: original tokens
            # y: juman tokens
            x2y, _ = get_alignments(tokens, tokens_juman)
            annotations = []
            for label, start, end in tags_to_entities(tags):
                annotations.append(
                    {"start": x2y[start][0], "end": x2y[end][-1] + 1, "type": label}
                )

            yield tokens_juman, annotations


def main(input_path: str, output_path: str, lang: str) -> None:

    nlp = spacy.blank(lang)
    db = DocBin()

    for tokens_juman, annotations_juman in load_from_conllu(input_path):
        tokens, annotations = converter(tokens_juman, annotations_juman)
        tags = ["O"] * len(tokens)
        for start, end, entity in annotations:
            if any(tag != "O" for tag in tags[start:end]):
                continue
            if start + 1 == end:
                tags[start:end] = [f"U-{entity}"]
                continue
            tags[start:end] = (
                [f"B-{entity}"] + [f"I-{entity}"] * (end - start - 2) + [f"L-{entity}"]
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
    parser.add_argument("--lang", type=str, default="ja", help="Language")
    parser.add_argument("--input", type=str, required=True, help="An input file path")
    parser.add_argument("--output", type=str, default=None, help="An output file path")

    args = parser.parse_args()

    if args.output is None:
        basename, _ = os.path.splitext(args.input)
        output = f"{basename}.spacy"
    else:
        output = args.output

    main(args.input, output, args.lang)
