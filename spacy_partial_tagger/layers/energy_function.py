import torch
from partial_tagger.crf import CRF
from torch import nn
from transformers import AutoModel, BatchEncoding


class PartialTransformerEnergyFunction(nn.Module):
    def __init__(
        self, model_name: str, feature_size: int, num_tags: int, dropout: float = 0.2
    ) -> None:
        super(PartialTransformerEnergyFunction, self).__init__()

        self.feature_extractor = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(feature_size, num_tags)

    def forward(self, batch: BatchEncoding) -> torch.Tensor:
        text_features = self.feature_extractor(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            token_type_ids=batch.token_type_ids,
        ).last_hidden_state
        return self.crf(self.dropout(text_features), batch.attention_mask.bool())
