import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import spacy
from spacy.tokens import Doc, DocBin
from spacy.training.iob_utils import biluo_tags_to_spans
from transformers import AutoTokenizer


def main(input_path: str, output_path: str, model_name: str, lang: str) -> None:

    nlp = spacy.blank(lang)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    db = DocBin()

    with open(input_path) as f:
        for data in map(json.loads, f):
            tokenized_data = tokenizer(  # type:ignore
                " ".join(data["tokens"]),
                add_special_tokens=True,
                return_token_type_ids=False,
                return_attention_mask=False,
                return_offsets_mapping=True,
            )
            tokens = tokenized_data["input_ids"]
            charidx2tokidx: dict = {}
            for i, t in enumerate(data["tokens"]):
                offset = len(charidx2tokidx) + i
                for k in range(len(t)):
                    charidx2tokidx[offset + k] = i

            # Mapping derived from the offset mapping to go brom token idx to bpe idx
            offset_mapping = tokenized_data.pop("offset_mapping")
            tokidx2bpeidx = defaultdict(list)
            for i, (s, e) in enumerate(offset_mapping):
                if s == e:
                    continue  # skip special tokens
                tokidx = charidx2tokidx[s]
                tokidx2bpeidx[tokidx].append(i)
            annotations = [
                # [start, end)
                (
                    tokidx2bpeidx[annotation["start"]][0],
                    tokidx2bpeidx[annotation["end"] - 1][-1] + 1,
                    annotation["type"],
                )
                for annotation in data["gold_annotations"]
            ]
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
                words=tokenizer.convert_ids_to_tokens(tokens),
                spaces=[False] * len(tokens),
            )
            ents = []
            for span in biluo_tags_to_spans(doc, tags):
                ent = doc.char_span(span.start_char, span.end_char, span.label_)
                ents.append(ent)
            doc.ents = ents  # type:ignore
            db.add(doc)

    db.to_disk(output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", help="Language")
    parser.add_argument(
        "--model", type=str, default="roberta-base", help="Transformer model"
    )
    parser.add_argument("--input", type=str, required=True, help="An input file path")
    parser.add_argument("--output", type=str, default=None, help="An output file path")

    args = parser.parse_args()

    if args.output is None:
        basename, _ = os.path.splitext(args.input)
        output = f"{basename}.spacy"
    else:
        output = args.output

    main(args.input, output, args.model, args.lang)
