import json
from typing import List

import catalogue
from conllu import parse_incr
from spacy.training.iob_utils import tags_to_entities
from spacy.util import registry
from tokenizations import get_alignments as get_alignments_original

registry.label_indexers = catalogue.create(  # type:ignore
    "spacy", "label_indexers", entry_points=True
)


def get_alignments(source: List[str], target: List[str]) -> List[List[int]]:
    _, y2x = get_alignments_original(source, target)
    indices = iter(range(len(source)))
    for y in y2x:
        if not y:
            # TODO: Fix me, maybe this doesn't work properly when [UNK] corresponds
            # more than 1 letter.
            for i in indices:
                if source[i] != " ":
                    y.append(i)
                    break
        else:
            for j in y:
                if j in indices:
                    continue
    return y2x


def to_jsonl(input_path: str, output_path: str) -> None:

    outfile = open(output_path, "w")

    with open(input_path) as infile:
        for i, data in enumerate(parse_incr(infile)):
            tags = ["O"] * len(data)
            tokens = []
            for i, x in enumerate(data):
                if "NE" in x["misc"]:
                    tags[i] = x["misc"]["NE"]
                tokens.append(x["form"])
            annotations = []
            for label, start, end in tags_to_entities(tags):
                annotations.append(
                    {
                        "start": start,
                        "end": end + 1,
                        "type": label,
                        "kind": "entity",
                        "mention": " ".join(tokens[start : end + 1]),
                    }
                )
            new_data = {
                "uid": f"{i}",
                "tokens": tokens,
                "gold_annotations": annotations,
                "is_complete": True,
            }

            data_str = json.dumps(new_data)
            outfile.write(f"{data_str}\n")

    outfile.close()
