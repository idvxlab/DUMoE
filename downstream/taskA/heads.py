import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Task A ， frozen embedding 。"""

    def __init__(
        self,
        d_in: int,
        num_classes: int = 15,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
