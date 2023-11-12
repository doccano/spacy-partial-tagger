from typing import Callable, Dict, Iterable, List, Optional, Tuple, cast

import srsly
import torch
from partial_tagger.crf import functional as F
from partial_tagger.training import create_tag_bitmap
from sequence_label import LabelSet, SequenceLabel
from spacy import util
from spacy.errors import Errors
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
from thinc.api import torch2xp, xp2torch
from thinc.config import Config
from thinc.model import Model
from thinc.optimizers import Optimizer
from thinc.types import Floats2d, Floats4d


def compute_partially_supervised_loss(
    log_potentials: torch.Tensor,
    tag_bitmap: torch.Tensor,
    mask: torch.Tensor,
    outside_index: int,
    target_entity_ratio: float = 0.15,
    entity_ratio_margin: float = 0.05,
    balancing_coefficient: int = 10,
) -> torch.Tensor:
    with torch.enable_grad():
        # log partition
        log_Z = F.forward_algorithm(log_potentials)

        # marginal probabilities
        p = torch.autograd.grad(log_Z.sum(), log_potentials, create_graph=True)[0].sum(
            dim=-1
        )
    p *= mask[..., None]

    expected_entity_count = (
        p[:, :, :outside_index].sum() + p[:, :, outside_index + 1 :].sum()
    )
    expected_entity_ratio = expected_entity_count / p.sum()
    expected_entity_ratio_loss = torch.clamp(
        (expected_entity_ratio - target_entity_ratio).abs() - entity_ratio_margin,
        min=0,
    )

    score = F.multitag_sequence_score(log_potentials, tag_bitmap, mask)
    supervised_loss = (log_Z - score).mean()

    return supervised_loss + balancing_coefficient * expected_entity_ratio_loss


class PartialEntityRecognizer(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str,
        scorer: Callable,
        padding_index: int = -1,
        unknown_index: int = -100,
    ) -> None:
        self.vocab = vocab
        self.model = model
        self.name = name
        self.scorer = scorer
        self.padding_index = padding_index
        self.unknown_index = unknown_index
        self.cfg: Dict[str, List[str]] = {"labels": []}

    @property
    def label_set(self) -> LabelSet:
        return LabelSet(
            labels=set(self.cfg["labels"]), padding_index=self.padding_index
        )

    def predict(self, docs: List[Doc]) -> Floats2d:
        (_, tag_indices) = self.model.predict(docs)
        return tag_indices

    def set_annotations(
        self,
        docs: List[Doc],
        tag_indices: Floats2d,
    ) -> None:
        labels = self.label_set.decode(
            tag_indices=tag_indices.tolist(),
            alignments=tuple(doc.user_data["alignment"] for doc in docs),
        )

        for doc, label in zip(docs, labels):
            ents = []
            for tag in label.tags:
                span = doc.char_span(tag.start, tag.start + tag.length, tag.label)
                if span:
                    ents.append(span)
            if ents:
                doc.ents = tuple(ents)  # type:ignore

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> dict:
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)

        docs = [example.x for example in examples]
        (log_potentials, _), backward = self.model.begin_update(docs)

        loss, grad = self.get_loss(examples, log_potentials)
        backward((grad, None))

        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss
        return losses

    def initialize(self, get_examples: Callable, *, nlp: Language) -> None:
        X_small: List[Doc] = []
        labels: List[str] = []
        for example in get_examples():
            if len(X_small) < 10:
                X_small.append(example.x)
            for entity in example.y.ents:
                if entity.label_ not in labels:
                    labels.append(entity.label_)

        self.cfg["labels"] = labels

        self.model.initialize(
            X=X_small,
            Y=self.label_set,
        )

    def get_loss(
        self, examples: Iterable[Example], scores: Floats4d
    ) -> Tuple[float, Floats4d]:
        scores_pt = xp2torch(scores, requires_grad=True)

        labels = []
        alignments = []
        lengths = []
        for example in examples:
            labels.append(
                SequenceLabel.from_dict(
                    tags=[
                        {
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "label": ent.label_,
                        }
                        for ent in example.y.ents
                    ],
                    size=len(example.y.text),
                )
            )

            alignment = example.x.user_data["alignment"]
            alignments.append(alignment)
            lengths.append(alignment.target_size)

        tag_bitmap = create_tag_bitmap(
            label_set=self.label_set,
            labels=tuple(labels),
            alignments=tuple(alignments),
            device=scores_pt.device,
        )

        max_length = max(lengths)
        mask = torch.tensor(
            [[True] * length + [False] * (max_length - length) for length in lengths],
            device=scores_pt.device,
        )

        loss = compute_partially_supervised_loss(
            scores_pt, tag_bitmap, mask, self.label_set.outside_index
        )

        (grad,) = torch.autograd.grad(loss, scores_pt)

        return loss.item(), cast(Floats4d, torch2xp(grad))

    def add_label(self, label: str) -> int:
        if label in self.cfg["labels"]:
            return 0
        self.cfg["labels"].append(label)
        return 1

    def from_bytes(
        self, bytes_data: bytes, *, exclude: tuple = ()
    ) -> "PartialEntityRecognizer":
        self._validate_serialization_attrs()

        def load_model(b: bytes) -> None:
            try:
                self.model.from_bytes(b)
            except AttributeError:
                raise ValueError(Errors.E149) from None

        deserialize = {}
        if hasattr(self, "cfg") and self.cfg is not None:
            deserialize["cfg"] = lambda b: self.cfg.update(srsly.json_loads(b))
        deserialize["vocab"] = lambda b: self.vocab.from_bytes(  # type:ignore
            b, exclude=exclude
        )

        util.from_bytes(bytes_data, deserialize, exclude)
        self.model.initialize(Y=self.label_set)

        model_deserializers = {
            "model": lambda b: self.model.from_bytes(b),
        }
        util.from_bytes(bytes_data, model_deserializers, exclude)
        return self

    def from_disk(self, path: str, exclude: tuple = ()) -> "PartialEntityRecognizer":
        self._validate_serialization_attrs()

        def load_model(p: str) -> None:
            try:
                with open(p, "rb") as mfile:
                    self.model.from_bytes(mfile.read())
            except AttributeError:
                raise ValueError(Errors.E149) from None

        deserialize = {}
        if hasattr(self, "cfg") and self.cfg is not None:
            deserialize["cfg"] = lambda p: self.cfg.update(srsly.read_json(p))
        deserialize["vocab"] = lambda p: self.vocab.from_disk(  # type:ignore
            p, exclude=exclude
        )
        util.from_disk(path, deserialize, exclude)
        self.model.initialize(Y=self.label_set)
        model_deserializers = {
            "model": load_model,
        }
        util.from_disk(path, model_deserializers, exclude)  # type:ignore
        return self


default_model_config = """
[model]
@architectures = "spacy-partial-tagger.PartialTagger.v1"
transformer_model_name = "roberta-base"
padding_index = -1
"""
DEFAULT_NER_MODEL = Config().from_str(default_model_config)["model"]


@Language.factory(
    "partial_ner",
    assigns=["doc.ents", "token.ent_iob", "token.ent_type"],
    default_config={
        "model": DEFAULT_NER_MODEL,
        "scorer": {"@scorers": "spacy.ner_scorer.v1"},
    },
    default_score_weights={
        "ents_f": 1.0,
        "ents_p": 0.0,
        "ents_r": 0.0,
        "ents_per_type": None,
    },
)
def make_partial_ner(
    nlp: Language,
    name: str,
    model: Model,
    scorer: Callable,
    padding_index: int = -1,
    unknown_index: int = -100,
) -> PartialEntityRecognizer:
    return PartialEntityRecognizer(
        nlp.vocab, model, name, scorer, padding_index, unknown_index
    )
