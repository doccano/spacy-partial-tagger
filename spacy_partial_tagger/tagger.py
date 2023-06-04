from functools import partial
from typing import Any, Callable, List, Optional, Tuple, cast

from partial_tagger.data import LabelSet
from partial_tagger.data.batch.text import BaseTokenizer
from partial_tagger.utils import create_tagger
from spacy.tokens import Doc
from spacy.util import registry
from thinc.api import Model, get_torch_default_device, torch2xp, xp2torch
from thinc.shims import PyTorchGradScaler, PyTorchShim
from thinc.types import ArgsKwargs, Floats4d, Ints2d
from thinc.util import convert_recursive, is_torch_array, is_xp_array

from .tokenizer import get_tokenizer


@registry.architectures.register("spacy-partial-tagger.PartialTagger.v1")
def build_partial_tagger_v1(
    transformer_model_name: str,
    padding_index: int,
    tokenizer_args: Optional[dict] = None,
    *,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None,
) -> Model[List[Doc], Tuple[Floats4d, Ints2d]]:
    return Model(
        name="partial_tagger",
        forward=forward,
        init=init,
        attrs={
            "transformer_model_name": transformer_model_name,
            "padding_index": padding_index,
            "tokenizer_args": tokenizer_args,
            "mixed_precision": mixed_precision,
            "grad_scaler": grad_scaler,
        },
    )


def forward(
    model: Model[List[Doc], Tuple[Floats4d, Ints2d]],
    X: List[Doc],
    is_train: bool,
) -> Tuple[Tuple[Floats4d, Ints2d], Callable]:

    tokenizer: BaseTokenizer = model.attrs["tokenizer"]

    text_batch = tokenizer(tuple(doc.text for doc in X))

    for doc, text in zip(X, text_batch.tokenized_texts):
        doc.user_data["tokenized_text"] = text

    device = get_torch_default_device()

    (log_potentials, tag_indices), backward = model.layers[0](
        [text_batch.get_tagger_inputs(device), text_batch.get_mask(device)],
        is_train,
    )

    return (log_potentials, tag_indices), backward


def init(
    model: Model[List[Doc], Tuple[Floats4d, Ints2d]],
    X: List[Doc],
    Y: LabelSet,
) -> None:
    if model.layers:
        return

    transformer_model_name = model.attrs["transformer_model_name"]
    padding_index = model.attrs["padding_index"]
    tokenizer_args = model.attrs["tokenizer_args"]
    mixed_precision = model.attrs["mixed_precision"]
    grad_scaler = model.attrs["grad_scaler"]

    model.attrs["tokenizer"] = get_tokenizer(transformer_model_name, tokenizer_args)

    tagger = create_tagger(transformer_model_name, Y, padding_index)
    PyTorchWrapper = registry.get("layers", "PyTorchWrapper.v2")

    model._layers = [
        PyTorchWrapper(
            tagger,
            mixed_precision=mixed_precision,
            grad_scaler=grad_scaler,
            convert_outputs=convert_tagger_outputs,
        )
    ]


def convert_tagger_outputs(
    model: Model[List[Doc], Tuple[Floats4d, Ints2d]], X_Ytorch: Any, is_train: bool
) -> tuple:
    shim = cast(PyTorchShim, model.shims[0])
    X, Ytorch = X_Ytorch
    Y = convert_recursive(is_torch_array, torch2xp, Ytorch)

    def reverse_conversion(dY: Any) -> ArgsKwargs:
        dYtorch = convert_recursive(
            is_xp_array, partial(xp2torch, device=shim.device), dY
        )
        return ArgsKwargs(args=((Ytorch[0]),), kwargs={"grad_tensors": dYtorch[0]})

    return Y, reverse_conversion
