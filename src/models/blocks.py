# src/models/blocks.py
"""
Blocos reutilizáveis por todas as arquiteturas.
"""

import torch
import torch.nn as nn
from typing import List

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


class MLP(nn.Module):
    """
    MLP genérico com LayerNorm opcional na entrada.
    input_norm → Linear → Act → Dropout (×n) → Linear
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,
                 dropout=0.1, activation="relu", input_norm=True):
        super().__init__()
        act = ACTIVATIONS.get(activation, nn.ReLU)
        layers = []
        if input_norm:
            layers.append(nn.LayerNorm(input_dim))
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), act(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Adapter(nn.Module):
    """
    Projeta um embedding de dim_in → adapter_dim.
    Linear → LayerNorm? → Dropout
    """
    def __init__(self, dim_in: int, adapter_dim: int,
                 dropout: float = 0.1, use_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim_in, adapter_dim)]
        if use_norm:
            layers.append(nn.LayerNorm(adapter_dim))
        layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_adapters(emb_cfgs: list, adapter_cfg: dict) -> nn.ModuleList:
    """Cria um Adapter por embedding listado na config."""
    return nn.ModuleList([
        Adapter(
            dim_in      = e["dim"],
            adapter_dim = adapter_cfg["adapter_dim"],
            dropout     = adapter_cfg.get("dropout", 0.1),
            use_norm    = adapter_cfg.get("use_norm", True),
        )
        for e in emb_cfgs
    ])