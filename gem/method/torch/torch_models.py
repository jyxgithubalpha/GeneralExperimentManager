"""
PyTorch tabular models (MLP and FT-Transformer).
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    return nn.ReLU()


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [256, 128]
        layers = []
        prev = input_dim
        act = _activation(activation)
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def feature_importance(self) -> torch.Tensor:
        first = None
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                first = layer
                break
        if first is None:
            return torch.ones(self.input_dim, device=self.net[0].weight.device)
        return torch.norm(first.weight, dim=0)


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        return x.unsqueeze(-1) * self.weight + self.bias

    def importance(self) -> torch.Tensor:
        return torch.norm(self.weight, dim=1)


class FTTransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, d_token)
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        if self.use_cls_token:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        pooled = encoded[:, 0] if self.use_cls_token else encoded.mean(dim=1)
        return self.head(pooled).squeeze(-1)

    def feature_importance(self) -> torch.Tensor:
        return self.tokenizer.importance()
