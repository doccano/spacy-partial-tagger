import os
import unicodedata
from argparse import ArgumentParser
from typing import Iterable, List, Tuple

import spacy
import unidic_lite
from conllu import parse_incr
from fugashi import GenericTagger
from mojimoji import han_to_zen
from pyknp import Juman
from spacy.tokens import Doc, DocBin
from spacy.training.iob_utils import biluo_tags_to_spans, tags_to_entities
from tokenizations import get_alignments

from .convert import converter

jumanpp = None
fugashi = None


def tokenize_jumanpp(text: str) -> List[str]:
    global jumanpp
    if jumanpp is None:
        jumanpp = Juman("etc/bin/jumanpp")
    tokens = []
    for mrph in jumanpp.analysis(text).mrph_list():
        if mrph.bunrui == "空白":
            continue
        tokens.append(mrph.midasi)
    return tokens


def tokenize_fugashi(text: str) -> List[str]:
    global fugashi
    if fugashi is None:
        dic_dir = unidic_lite.DICDIR
        mecabrc = os.path.join(dic_dir, "mecabrc")
        fugashi = GenericTagger("-d {} -r {} ".format(dic_dir, mecabrc))
    tokens = []
    cursor = 0
    for word in fugashi(text):
        token = word.surface
        start = str(text).index(token, cursor)
        end = start + len(token)

        tokens.append(text[start:end])
        cursor = end

    return tokens


def load_from_conllu_with_fugashi(path: str) -> Iterable[Tuple[List[str], List[dict]]]:
    with open(path) as f:
        for data in parse_incr(f):
            tags = ["O"] * len(data)
            text = ""
            tokens = []
            for i, x in enumerate(data):
                if "NE" in x["misc"]:
                    tags[i] = x["misc"]["NE"]
                tokens.append(x["form"])
                text += x["form"]
                if x["misc"]["SpaceAfter"] == "Yes":
                    text += " "
            tokens_fugashi = tokenize_fugashi(text)
            # x: original tokens
            # y: fugashi tokens
            x2y, _ = get_alignments(tokens, tokens_fugashi)
            annotations = []
            for label, start, end in tags_to_entities(tags):
                assert "".join(tokens[start : end + 1]) in "".join(
                    tokens_fugashi[x2y[start][0] : x2y[end][-1] + 1]
                )
                annotations.append(
                    {"start": x2y[start][0], "end": x2y[end][-1] + 1, "type": label}
                )

            tokens_fugashi = [
                unicodedata.normalize("NFKC", token) for token in tokens_fugashi
            ]
            yield tokens_fugashi, annotations


def load_from_conllu_with_jumanpp(
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
                    text += " "
            text = han_to_zen(text)
            tokens_juman = tokenize_jumanpp(text)
            # x: original tokens
            # y: juman tokens
            x2y, _ = get_alignments(tokens, tokens_juman)
            annotations = []
            for label, start, end in tags_to_entities(tags):
                assert "".join(tokens[start : end + 1]) in "".join(
                    tokens_juman[x2y[start][0] : x2y[end][-1] + 1]
                )
                annotations.append(
                    {"start": x2y[start][0], "end": x2y[end][-1] + 1, "type": label}
                )

            yield tokens_juman, annotations


def load_from_conllu(path: str) -> Iterable[Tuple[List[str], List[dict]]]:
    with open(path) as f:
        for data in parse_incr(f):
            tags = ["O"] * len(data)
            tokens = []
            for i, x in enumerate(data):
                if "NE" in x["misc"]:
                    tags[i] = x["misc"]["NE"]
                tokens.append(x["form"])
            annotations = []
            for label, start, end in tags_to_entities(tags):
                annotations.append({"start": start, "end": end + 1, "type": label})
            yield tokens, annotations


def main(input_path: str, output_path: str, lang: str) -> None:

    nlp = spacy.blank(lang)
    db = DocBin()

    mode = "fugashi"
    if mode == "fugashi":
        loader = load_from_conllu_with_fugashi
    elif mode == "jumanpp":
        loader = load_from_conllu_with_jumanpp
    else:
        loader = load_from_conllu

    for tokens_, annotations_ in loader(input_path):
        tokens, annotations = converter(tokens_, annotations_)
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
