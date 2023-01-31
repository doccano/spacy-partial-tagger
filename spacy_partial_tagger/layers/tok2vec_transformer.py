from typing import Any, Optional, Tuple, cast

import torch
from spacy.tokens import Doc
from spacy.util import List, registry
from thinc.api import ArgsKwargs, Model, get_torch_default_device, torch2xp, xp2torch
from thinc.shims.pytorch_grad_scaler import PyTorchGradScaler
from thinc.types import Floats2d, Floats3d
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer, BatchEncoding, BertJapaneseTokenizer

from ..aligners import TransformerAligner
from ..util import get_alignments
from .constrainer import Constrainer, ConstrainerFactory, get_token_mapping


class TransformersWrapper(Module):
    def __init__(self, transformer: Module) -> None:
        super(TransformersWrapper, self).__init__()

        self.transformer = transformer

    def forward(self, inputs: BatchEncoding) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = inputs.attention_mask.sum(dim=-1)
        outputs = self.transformer(**inputs).last_hidden_state
        return outputs, lengths


@registry.architectures.register("spacy-partial-tagger.MisalignedTok2VecTransformer.v1")
def build_misaligned_tok2vec_transformer(
    model_name: str,
    chunk_size: int = 0,
    max_length: Optional[int] = None,
    *,
    mixed_precision: bool = False,
    grad_scaler: Optional[PyTorchGradScaler] = None
) -> Model[List[Doc], Tuple[List[Floats2d], List[TransformerAligner], Constrainer]]:
    return Model(
        "misaligned_tok2vec_transformer",
        forward=forward,
        init=init,
        dims={"nI": None, "nO": None},
        attrs={
            "model_name": model_name,
            "chunk_size": chunk_size,
            "max_length": max_length,
            "mixed_precision": mixed_precision,
            "grad_scaler": grad_scaler,
        },
    )


def init(model: Model, X: Any = None, Y: dict = {0: "O"}) -> None:
    if model.layers:
        return

    model_name = model.attrs["model_name"]

    model.attrs["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
    model.attrs["constrainer_factory"] = ConstrainerFactory(Y)

    PyTorchWrapper = registry.get("layers", "PyTorchWrapper.v2")

    mixed_precision = model.attrs["mixed_precision"]
    grad_scaler = model.attrs["grad_scaler"]

    pytorch_model = AutoModel.from_pretrained(
        model_name, chunk_size_feed_forward=model.attrs["chunk_size"]
    )

    transformer = PyTorchWrapper(
        TransformersWrapper(pytorch_model),
        mixed_precision=mixed_precision,
        convert_outputs=convert_transformer_outputs,
        grad_scaler=grad_scaler,
    )
    model.set_dim("nO", pytorch_model.config.hidden_size)
    model._layers = [transformer]


def forward(model: Model, X: Any, is_train: bool) -> tuple:
    tokenizer = model.attrs["tokenizer"]
    constrainer_factory = model.attrs["constrainer_factory"]
    texts = [doc.text for doc in X]
    device = get_torch_default_device()
    max_length = model.attrs["max_length"]
    if max_length is not None:
        padding = "max_length"
    else:
        padding = "longest"
    X_transformers = tokenizer(
        texts,
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors="pt",
        return_offsets_mapping=tokenizer.is_fast,
        padding=padding,
        max_length=max_length,
        truncation=False,
    ).to(device)

    if tokenizer.is_fast:
        lengths = X_transformers["attention_mask"].sum(dim=-1)
        mappings = [
            [list(range(start, end)) for start, end in mapping[: lengths[i]]]
            for i, mapping in enumerate(X_transformers.pop("offset_mapping"))
        ]
    elif isinstance(tokenizer, BertJapaneseTokenizer):
        mappings = [
            get_alignments(tokenizer, text, ids)
            for text, ids in zip(texts, X_transformers.input_ids.tolist())
        ]
    else:
        raise ValueError("Ordinary Tokenizers are not supported.")

    lengths = [len(text) for text in texts]
    aligners = [
        TransformerAligner(mapping, length)
        for mapping, length in zip(mappings, lengths)
    ]
    constrainer = constrainer_factory.get_constrainer(
        [get_token_mapping(doc, mapping) for doc, mapping in zip(X, mappings)]
    )
    Y, backward = model.layers[0](X_transformers, is_train)
    return (Y, aligners, constrainer), backward


def convert_transformer_outputs(
    model: Model, inputs_outputs: tuple, is_train: bool
) -> tuple:
    pad = model.ops.pad
    unpad = model.ops.unpad

    _, (Yt, Lt) = inputs_outputs

    def convert_for_torch_backward(
        dY: Tuple[List[Floats2d], List[TransformerAligner], Constrainer]
    ) -> ArgsKwargs:
        # Ignore gradients for aligners
        dY_t = xp2torch(pad(dY[0], round_to=Yt.size(1)))
        return ArgsKwargs(args=(Yt,), kwargs={"grad_tensors": dY_t})  # type:ignore

    Y = cast(Floats3d, torch2xp(Yt))
    return (
        cast(List[Floats2d], unpad(Y, Lt.tolist())),
        convert_for_torch_backward,
    )
