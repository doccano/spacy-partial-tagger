from typing import Callable, Iterable, List

import srsly
from spacy import util
from spacy.errors import Errors
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc, Span
from spacy.training import Example, biluo_tags_to_spans, iob_to_biluo
from spacy.vocab import Vocab
from thinc.config import Config
from thinc.model import Model
from thinc.optimizers import Optimizer

from spacy_partial_tagger.loss import ExpectedEntityRatioLoss
from spacy_partial_tagger.util import from_subword_tags, to_subword_tags


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
        self.cfg: dict = {
            "labels": [],
            "tag_to_id": {},
            "id_to_tag": {},
            "outside_index": 0,
            "padding_index": padding_index,
            "unknown_index": unknown_index,
        }

    @property
    def labels(self) -> list:
        return self.cfg["labels"]

    @property
    def tag_to_id(self) -> dict:
        return self.cfg["tag_to_id"]

    @property
    def id_to_tag(self) -> list:
        return self.cfg["id_to_tag"]

    @property
    def outside_index(self) -> int:
        return self.cfg["outside_index"]

    @property
    def padding_index(self) -> int:
        return self.cfg["padding_index"]

    @property
    def unknown_index(self) -> int:
        return self.cfg["unknown_index"]

    @staticmethod
    def _to_sentences(doc: Doc) -> List[Doc]:
        if doc.has_annotation("SENT_START"):
            return [sent.as_doc() for sent in doc.sents]
        else:
            return [doc]

    def predict(self, docs: List[Doc]) -> tuple:
        docs = [sent for doc in docs for sent in self._to_sentences(doc)]
        _, guesses, offset_mappings = self.model.predict(docs)
        return guesses, offset_mappings

    def set_annotations(self, docs: List[Doc], batch_tag_indices: tuple) -> None:
        guesses, offset_mappings = (
            batch_tag_indices[0].tolist(),
            batch_tag_indices[1].tolist(),
        )
        index = 0
        for doc in docs:
            ents = []
            offset = 0
            for sent in self._to_sentences(doc):
                subword_tags = []
                for i in guesses[index]:
                    if i == -1:
                        break
                    subword_tags.append(self.id_to_tag[i])  # type:ignore
                tags = from_subword_tags(
                    subword_tags, offset_mappings[index], len(sent)
                )
                for span in biluo_tags_to_spans(sent, tags):
                    ents.append(
                        Span(
                            doc,
                            span.start + offset,
                            span.end + offset,
                            label=span.label,
                        )
                    )
                index += 1
                offset += len(tags)
            doc.ents = ents  # type:ignore

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optimizer = None,
        losses: dict = None,
    ) -> dict:
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        (log_potentials, _, offset_mapping), backward = self.model.begin_update(
            [example.x for example in examples]
        )
        loss, grad = self.get_loss(examples, (log_potentials, offset_mapping))
        backward(grad)
        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss
        return losses

    def initialize(
        self, get_examples: Callable, *, nlp: Language, labels: dict = None
    ) -> None:
        tag_to_id: dict = {"O": 0}
        id_to_tag: dict = {0: "O"}
        for example in get_examples():
            tags = []
            for token in example.y:
                if token.ent_iob_ != "O":
                    tag = f"{token.ent_iob_}-{token.ent_type_}"
                else:
                    tag = token.ent_iob_
                tags.append(tag)
            for tag in iob_to_biluo(tags):
                if tag not in tag_to_id:
                    id_to_tag[len(tag_to_id)] = tag
                    tag_to_id[tag] = len(tag_to_id)

        for tag in tag_to_id:
            if tag == "O":
                self.add_label(tag)
            else:
                self.add_label(tag.split("-")[1])
        self.cfg["tag_to_id"] = tag_to_id
        self.cfg["id_to_tag"] = id_to_tag
        self.cfg["outside_index"] = tag_to_id["O"]

        self.model.initialize(Y=id_to_tag)

    def get_loss(self, examples: Iterable[Example], scores: tuple) -> tuple:
        potentials, offset_mappings = scores
        padding_index = self.padding_index
        unknown_index = self.unknown_index
        outside_index = self.outside_index
        loss_func = ExpectedEntityRatioLoss(padding_index, unknown_index, outside_index)
        truths = []
        for example, offset_mapping in zip(examples, offset_mappings.tolist()):
            tags = iob_to_biluo(
                [
                    f"{token.ent_iob_}-{token.ent_type_}"
                    if token.ent_iob_ != "O"
                    else "O"
                    for token in example.y
                ]
            )
            subword_tags = to_subword_tags(tags, offset_mapping)
            tag_indices = [
                self.tag_to_id[tag] if tag != "O" else unknown_index
                for tag in subword_tags
            ]
            # start/end token
            tag_indices[0] = tag_indices[-1] = self.tag_to_id["O"]
            truths.append(tag_indices)
        max_length = max(map(len, truths))
        truths = self.model.ops.asarray(  # type:ignore
            [
                truth + [padding_index] * (max_length - len(truth)) for truth in truths
            ]  # type:ignore
        )
        return loss_func(potentials, truths)[::-1]  # type:ignore

    def add_label(self, label: str) -> int:
        if label in self.labels:
            return 0
        self.labels.append(label)
        self.vocab.strings.add(label)
        return 1

    def from_bytes(
        self, bytes_data: bytes, *, exclude: tuple = tuple()
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

        self.model.initialize(Y=self.id_to_tag)

        self.cfg["id_to_tag"] = {
            int(key): value for key, value in self.cfg["id_to_tag"].items()
        }
        model_deserializers = {
            "model": lambda b: self.model.from_bytes(b),
        }
        util.from_bytes(bytes_data, model_deserializers, exclude)
        return self

    def from_disk(
        self, path: str, exclude: tuple = tuple()
    ) -> "PartialEntityRecognizer":
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
        self.cfg["id_to_tag"] = {
            int(key): value for key, value in self.cfg["id_to_tag"].items()
        }
        self.model.initialize(Y=self.id_to_tag)
        model_deserializers = {
            "model": load_model,
        }
        util.from_disk(path, model_deserializers, exclude)  # type:ignore
        return self


default_model_config = """
[model]
@architectures = "spacy-partial-tagger.PartialTransformerTagger.v1"
model_name = "roberta-base"
feature_size = 768
"""
DEFAULT_NER_MODEL = Config().from_str(default_model_config)["model"]


@Language.factory(
    "partial_ner",
    assigns=["doc.ents", "token.ent_iob", "token.ent_type"],
    default_config={
        "model": DEFAULT_NER_MODEL,
        "scorer": {"@scorers": "spacy.ner_scorer.v1"},
        "padding_index": -1,
        "unknown_index": -100,
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
    padding_index: int,
    unknown_index: int,
) -> PartialEntityRecognizer:
    return PartialEntityRecognizer(nlp.vocab, model, name, scorer)
