from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class UserRepModel(nn.Module):
    """（ + encode）。"""

    @abstractmethod
    def forward(self, batch: dict) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        seq_ids: torch.Tensor,
        seq_mask: torch.Tensor,
        profile: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError
