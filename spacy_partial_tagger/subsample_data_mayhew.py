""" Given data files in standard format,
subsample them by first subsampling surface-mention groups.
This implements the sampling strategy
from Mayhew 19 "NER with Partially Annotated Training Data".
In our paper this is called "Non-native Speaker" (NNS) annotation

We do this by first downsampling via surface mentions until the target recall is hit,
then we add back random annotation spans to adjust the precision levels,
given the new tp count from the low-recall set.

If inpath = 'data/train.jsonl', recall=0.5, precision=0.9
then the outputs will be:
  'data/train_r0.5_p0.9.jsonl'
"""

import json
import logging
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from copy import deepcopy
from typing import Sequence

import colorlog
import numpy.random as npr
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TqdmHandler(logging.StreamHandler):
    def __init__(self) -> None:
        logging.StreamHandler.__init__(self)

    def emit(self, record) -> None:  # type:ignore
        msg = self.format(record)
        tqdm.write(msg)


handler = TqdmHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%d-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "SUCCESS:": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)
logger.addHandler(handler)


def parse_args(args: Sequence[str] = None) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("inpath", type=str)
    parser.add_argument("--recall", type=float, default=0.5)
    parser.add_argument("--precision", type=float, default=0.9)
    parser.add_argument("--kind", type=str, default="entity")

    parser.add_argument(
        "--limit", type=int, default=None, help="Limit to first N sentences"
    )
    parser.add_argument("--loglevel", type=str, default="INFO")

    parser.add_argument("--random-seed", type=int, default=42)

    if args:
        return parser.parse_args(args)
    else:
        return parser.parse_args()


def run(args: Namespace) -> None:
    logger.setLevel(args.loglevel)
    logger.info(f"Args: {args}")

    npr.seed(args.random_seed)

    outprefix = args.inpath[: -len(".jsonl")]

    # Read in the data and make sure we restrict to the right kind of annotation
    logger.info("Loading data")
    data = [json.loads(line) for line in open(args.inpath)]
    for i, d in enumerate(data):
        try:
            d["gold_annotations"] = [
                a for a in d["gold_annotations"] if a["kind"] == args.kind
            ]
        except Exception as e:
            logger.exception(f"Bad datum: {d} ... {e}")
            raise e

    # Collect mapping of annotation mentions to (datum_idx, annotation)
    logger.info("Sample partially-supervised data")
    annotations_by_mention = defaultdict(list)
    types = set()
    for i, d in enumerate(data):
        for a in d["gold_annotations"]:
            annotations_by_mention[a["mention"]].append((i, a))
            types.add(a["type"])

    def nann(annotations_by_mention: dict) -> int:
        return sum(len(ms) for ms in annotations_by_mention.values())

    n = nann(annotations_by_mention)
    target_n = int(args.recall * n)
    logger.info(
        f"\n  {len(annotations_by_mention)} unique mentions, {n} total annotations"
        f"\n  downsampling to {args.recall} recall and {args.precision} precision"
        f"\n  => targeting {target_n} true annotations left w/ "
        f"{int((1-args.precision)*target_n)} additional fps"
    )

    # Copy og data as psl data and clear out annotations
    psl_data = deepcopy(data)
    for d in psl_data:
        d["gold_annotations"] = []
        d["is_complete"] = False

    # Now drop out mention entries until the count crosses target_n
    random_keys = list(annotations_by_mention.keys())
    npr.shuffle(random_keys)
    for drop in random_keys:
        L = len(annotations_by_mention.pop(drop))
        n -= L
        logger.debug(f"Dropping {drop} w/ {L} mentions. Now have {n}, want {target_n}")
        if n <= target_n:
            break

    # Fill the psl data in with these kept mentions
    for mention_annotations in annotations_by_mention.values():
        for (d_idx, a) in mention_annotations:
            psl_data[d_idx]["gold_annotations"].append(a)

    # Finally, adjust the precision with random (but non-overlapping) annotations
    # by picking sentences at random, then starts at random, then len uniform in 1,2,3,
    # rejecting if it overlaps another
    logger.info("Recall lowered, now lowering precision")
    recall_n = nann(annotations_by_mention)
    n_fp = int((1 - args.precision) * recall_n)
    fp_count = 0
    while fp_count < n_fp:
        d = npr.choice(psl_data, size=1)[0]
        observed_idxs = {
            i for a in d["gold_annotations"] for i in range(a["start"], a["end"])
        }
        unobserved_idxs = list(set(range(len(d["tokens"]))) - observed_idxs)
        if unobserved_idxs:
            s = int(npr.choice(unobserved_idxs, size=1)[0])
            e_choices = [
                e
                for e in range(s + 1, min(len(d["tokens"]), s + 4))
                if e not in observed_idxs
            ]
            if e_choices:
                e = int(npr.choice(e_choices, size=1)[0])  # type:ignore
                t = str(npr.choice(list(types), size=1)[0])
                m = " ".join(d["tokens"][s:e])
                fp = dict(
                    kind=args.kind,
                    type=t,
                    start=s,
                    end=e,
                    mention=m,
                    comment="False Positive",
                )
                logger.debug(f"Adding FP: {fp} to datum: {d['gold_annotations']}")
                d["gold_annotations"].append(fp)
                fp_count += 1
            else:
                logger.debug(f"No valid end found for position {s} in {d}")
        else:
            logger.debug(f"Datum is fully annotated: {d}")

    logger.info(
        f"Targeting {target_n} true annotations, {int((1-args.precision)*target_n)} "
        f"additional fps. Got {recall_n} and {fp_count}."
    )

    # Sort the annotations by start and write them out
    for d in psl_data:
        d["gold_annotations"] = sorted(d["gold_annotations"], key=lambda a: a["start"])
    with open(
        f"{outprefix}_r{args.recall:1.1f}_p{args.precision:1.1f}.jsonl", "w"
    ) as outf:
        for d in psl_data:
            outf.write(f"{json.dumps(d)}\n")

    # Do some sanity checks
    logger.info("Sanity Checks")
    assert len(data) == len(psl_data), "Dataset sizes are not same"

    logger.info("All done")


if __name__ == "__main__":
    run(parse_args())
