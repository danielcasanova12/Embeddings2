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


# ---------------------------------------------------------------------------
# Pooling Strategies (Exp 8)
# ---------------------------------------------------------------------------

class MeanPool(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B, T, D]
        if mask is not None:
            # mask: [B, T]
            mask = mask.unsqueeze(-1).float()
            return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return x.mean(dim=1)


class StatsPool(nn.Module):
    """Mean + Standard Deviation pooling -> [B, 2*D]"""
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            mu = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            var = ((x - mu.unsqueeze(1))**2 * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            std = torch.sqrt(var.clamp(min=1e-9))
        else:
            mu = x.mean(dim=1)
            std = x.std(dim=1)
        return torch.cat([mu, std], dim=-1)


class ASPPool(nn.Module):
    """Attentive Statistics Pooling."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B, T, D]
        weights = self.attention(x)  # [B, T, D]
        
        if mask is not None:
            weights = weights * mask.unsqueeze(-1).float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
            
        mu = torch.sum(x * weights, dim=1)
        var = torch.sum(x**2 * weights, dim=1) - mu**2
        std = torch.sqrt(var.clamp(min=1e-9))
        
        return torch.cat([mu, std], dim=-1)


def get_pooling_layer(name: str, input_dim: int = None):
    name = name.lower()
    if name == "mean":
        return MeanPool()
    if name == "stats":
        return StatsPool()
    if name == "asp":
        return ASPPool(input_dim)
    raise ValueError(f"Pooling unknown: {name}")


# ---------------------------------------------------------------------------
# Adapters (Exp 2 & 3)
# ---------------------------------------------------------------------------

class Adapter(nn.Module):
    """
    Projeta um embedding de dim_in → adapter_dim.
    Linear(d_original → k) → LayerNorm → ReLU → Linear(k → k)
    """
    def __init__(self, dim_in: int, adapter_dim: int,
                 dropout: float = 0.1, use_norm: bool = True):
        super().__init__()
        layers = [nn.Linear(dim_in, adapter_dim)]
        if use_norm:
            layers.append(nn.LayerNorm(adapter_dim))
        layers += [
            nn.ReLU(),
            nn.Linear(adapter_dim, adapter_dim),
            nn.Dropout(dropout)
        ]
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