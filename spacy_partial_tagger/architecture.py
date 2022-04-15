from typing import Any, List, Optional

from spacy.tokens import Doc
from spacy.util import registry
from thinc.model import Model
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from transformers import AutoTokenizer

from spacy_partial_tagger.layers.decoder import ConstrainedDecoder, get_constraints
from spacy_partial_tagger.layers.energy_function import PartialTransformerEnergyFunction


@registry.architectures.register("spacy-partial-tagger.PartialTransformerTagger.v1")
def build_partial_transformer_tagger(
    model_name: str,
    feature_size: int,
    num_tags: Optional[int] = None,
    dropout: float = 0.2,
    padding_index: int = -1,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None,
) -> Model:

    return Model(
        name="partial_transformer_tagger",
        forward=partial_transformer_tagger_forward,
        init=partial_transformer_tagger_init,
        attrs={
            "tokenizer": AutoTokenizer.from_pretrained(model_name),
            "model_name": model_name,
            "feature_size": feature_size,
            "num_tags": num_tags,
            "dropout": dropout,
            "padding_index": padding_index,
            "mixed_precision": mixed_precision,
            "grad_scaler": grad_scaler,
        },
    )


def partial_transformer_tagger_forward(
    model: Model, docs: List[Doc], is_train: bool
) -> tuple:
    tokenizer = model.attrs["tokenizer"]

    texts = [
        tokenizer.decode(
            tokenizer.convert_tokens_to_ids([token.text for token in doc]),
            clean_up_tokenization_spaces=False,
        )
        for doc in docs
    ]
    X = tokenizer(
        texts,
        add_special_tokens=False,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    energy, backward = model.get_ref("energy_function")(X, is_train)

    tag_indices, _ = model.get_ref("decoder")(
        (energy, X.attention_mask.bool()), is_train
    )

    return (energy, tag_indices), backward


def partial_transformer_tagger_init(
    model: Model, X: Any = None, Y: dict = None
) -> None:
    if model.layers:
        return

    if Y is None:
        Y = {0: "O"}

    model_name = model.attrs["model_name"]
    feature_size = model.attrs["feature_size"]
    num_tags = model.attrs["num_tags"] or len(Y)
    dropout = model.attrs["dropout"]
    padding_index = model.attrs["padding_index"]
    mixed_precision = model.attrs["mixed_precision"]
    grad_scaler = model.attrs["grad_scaler"]

    PyTorchWrapper = registry.get("layers", "PyTorchWrapper.v2")

    energy_function = PyTorchWrapper(
        PartialTransformerEnergyFunction(
            model_name, feature_size, num_tags, dropout=dropout
        ),
        mixed_precision=mixed_precision,
        grad_scaler=grad_scaler,
    )
    decoder = PyTorchWrapper(
        ConstrainedDecoder(
            *get_constraints(Y),
            padding_index=padding_index,
        ),
        mixed_precision=mixed_precision,
        grad_scaler=grad_scaler,
    )

    model._layers = [energy_function, decoder]
    model.set_ref("energy_function", energy_function)
    model.set_ref("decoder", decoder)
