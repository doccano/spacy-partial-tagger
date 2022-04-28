from typing import Optional

import torch
from partial_tagger.crf.layer import CRF as _CRF
from torch import nn


class CRF(nn.Module):
    def __init__(
        self, embedding_size: int, num_tags: int, dropout: float = 0.2
    ) -> None:
        super(CRF, self).__init__()

        self.crf = _CRF(embedding_size, num_tags)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, embeddings: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is None:
            mask = embeddings.new_ones(embeddings.shape[:-1], dtype=torch.bool)
        else:
            mask = (
                torch.arange(
                    embeddings.size(1),
                    device=embeddings.device,
                )[None, :]
                < lengths[:, None]
            )
        return self.crf(self.dropout(embeddings), mask)
