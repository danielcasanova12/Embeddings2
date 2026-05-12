# src/models/architectures.py
"""
Implementação das 5 arquiteturas de predição de MOS.

Todas recebem como forward():
    embs: List[Tensor[B, D_i]]  — um tensor por embedding
    retornam: dict com pelo menos {"mos": Tensor[B]}
"""

import torch
import torch.nn as nn
from typing import List
from .blocks import MLP, Adapter, build_adapters


# ─────────────────────────────────────────────────────────────────────────────
# 1. Single Embedding
# ─────────────────────────────────────────────────────────────────────────────

class SingleEmbeddingMOS(nn.Module):
    """
    Audio → Embedding → Adapter → MLP → MOS
    Usa apenas o primeiro (e único) embedding da lista.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        emb_cfg = cfg["embeddings"][0]
        adp_cfg = cfg["adapter"]
        mlp_cfg = cfg["mlp"]

        self.adapter = Adapter(
            dim_in      = emb_cfg["dim"],
            adapter_dim = adp_cfg["adapter_dim"],
            dropout     = adp_cfg.get("dropout", 0.1),
            use_norm    = adp_cfg.get("use_norm", True),
        )
        self.mlp = MLP(
            input_dim  = adp_cfg["adapter_dim"],
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg["output_dim"],
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
        )

    def forward(self, embs: List[torch.Tensor]) -> dict:
        x = self.adapter(embs[0])
        return {"mos": self.mlp(x)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cross-Embedding Interaction
# ─────────────────────────────────────────────────────────────────────────────

class CrossEmbeddingInteraction(nn.Module):
    """
    Embeddings → Adapters → emb_i ⊗ emb_j (hadamard) para todos os pares → concat → MLP → MOS
    N embeddings geram N*(N-1)/2 pares de interação.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        emb_cfgs = cfg["embeddings"]
        adp_cfg  = cfg["adapter"]
        mlp_cfg  = cfg["mlp"]
        D        = adp_cfg["adapter_dim"]
        N        = len(emb_cfgs)
        n_pairs  = N * (N - 1) // 2

        self.adapters = build_adapters(emb_cfgs, adp_cfg)

        # Input do MLP: pares (hadamard) + embeddings individuais
        input_dim = D * N + D * n_pairs

        self.mlp = MLP(
            input_dim  = input_dim,
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg["output_dim"],
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
        )

    def forward(self, embs: List[torch.Tensor]) -> dict:
        adapted = [adp(e) for adp, e in zip(self.adapters, embs)]
        # Pares: hadamard product
        pairs = [
            adapted[i] * adapted[j]
            for i in range(len(adapted))
            for j in range(i + 1, len(adapted))
        ]
        x = torch.cat(adapted + pairs, dim=-1)
        return {"mos": self.mlp(x)}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-Embedding Concat Fusion
# ─────────────────────────────────────────────────────────────────────────────

class ConcatFusionMOS(nn.Module):
    """
    Embeddings → Adapters → Concatenação → MLP → MOS
    """
    def __init__(self, cfg: dict):
        super().__init__()
        emb_cfgs = cfg["embeddings"]
        adp_cfg  = cfg["adapter"]
        mlp_cfg  = cfg["mlp"]
        D        = adp_cfg["adapter_dim"]

        self.adapters = build_adapters(emb_cfgs, adp_cfg)
        self.mlp = MLP(
            input_dim  = D * len(emb_cfgs),
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg["output_dim"],
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
        )

    def forward(self, embs: List[torch.Tensor]) -> dict:
        adapted = [adp(e) for adp, e in zip(self.adapters, embs)]
        x = torch.cat(adapted, dim=-1)
        return {"mos": self.mlp(x)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reliability Fusion + Perceptual Factor Modeling
# ─────────────────────────────────────────────────────────────────────────────

class ReliabilityFusionMOS(nn.Module):
    """
    Embeddings → Adapters
        → Reliability estimation  (w_i = sigmoid(Linear(emb_i)))
        → Weighted sum
        → Factor heads            (predições auxiliares por fator perceptual)
        → MOS head
    
    A loss durante treino combina:
        L_total = L_mos + factor_weight * mean(L_factor_i)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        emb_cfgs     = cfg["embeddings"]
        adp_cfg      = cfg["adapter"]
        mlp_cfg      = cfg["mlp"]
        rel_cfg      = cfg.get("reliability", {})
        D            = adp_cfg["adapter_dim"]
        factor_names = rel_cfg.get("factor_names", [])

        self.adapters = build_adapters(emb_cfgs, adp_cfg)

        # Reliability: escalar por embedding, estimado do próprio embedding
        self.reliability_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(D, 1), nn.Sigmoid())
            for _ in emb_cfgs
        ])

        # MOS head
        self.mos_mlp = MLP(
            input_dim  = D,
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg["output_dim"],
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
        )

        # Factor heads (um por fator perceptual, predição auxiliar)
        self.factor_names = factor_names
        self.factor_heads = nn.ModuleList([
            nn.Linear(D, 1) for _ in factor_names
        ])

    def forward(self, embs: List[torch.Tensor]) -> dict:
        adapted = [adp(e) for adp, e in zip(self.adapters, embs)]  # [B, D] cada

        # Reliability weights: [B, N] → softmax para somar 1
        weights = torch.cat(
            [rh(a) for rh, a in zip(self.reliability_heads, adapted)],
            dim=-1
        )  # [B, N]
        weights = torch.softmax(weights, dim=-1)

        # Média ponderada: [B, D]
        stacked = torch.stack(adapted, dim=1)          # [B, N, D]
        fused   = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        mos = self.mos_mlp(fused)

        # Factor scores (usados na loss auxiliar durante treino)
        factor_scores = {
            name: self.factor_heads[i](fused).squeeze(-1)
            for i, name in enumerate(self.factor_names)
        }

        return {"mos": mos, "factors": factor_scores, "weights": weights}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Transformer Fusion
# ─────────────────────────────────────────────────────────────────────────────

class TransformerFusionMOS(nn.Module):
    """
    Embeddings → Adapters → [CLS, e1, e2, ... eN] → Transformer Encoder → CLS → MLP → MOS
    
    O token CLS é aprendido e serve como representação global.
    O Transformer aprende a cruzar informação entre embeddings heterogêneos.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        emb_cfgs = cfg["embeddings"]
        adp_cfg  = cfg["adapter"]
        mlp_cfg  = cfg["mlp"]
        tr_cfg   = cfg.get("transformer", {})
        D        = adp_cfg["adapter_dim"]
        N        = len(emb_cfgs)

        self.adapters = build_adapters(emb_cfgs, adp_cfg)

        # Token CLS aprendível
        self.cls_token = nn.Parameter(torch.randn(1, 1, D))

        # Positional embedding por posição (0=CLS, 1..N=embeddings)
        self.pos_emb = nn.Embedding(N + 1, D)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = D,
            nhead           = tr_cfg.get("num_heads", 4),
            dim_feedforward = tr_cfg.get("feedforward_dim", 512),
            dropout         = tr_cfg.get("dropout", 0.1),
            batch_first     = True,
            norm_first      = True,          # pre-norm (mais estável)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = tr_cfg.get("num_layers", 2),
        )

        # MLP final sobre o CLS
        self.mlp = MLP(
            input_dim  = D,
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg["output_dim"],
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
        )

    def forward(self, embs: List[torch.Tensor]) -> dict:
        B = embs[0].shape[0]
        adapted = [adp(e) for adp, e in zip(self.adapters, embs)]  # cada [B, D]

        # Sequência: [CLS, emb_0, emb_1, ..., emb_N-1]
        tokens = torch.stack(adapted, dim=1)                        # [B, N, D]
        cls    = self.cls_token.expand(B, -1, -1)                   # [B, 1, D]
        seq    = torch.cat([cls, tokens], dim=1)                    # [B, N+1, D]

        # Positional embeddings
        pos = torch.arange(seq.shape[1], device=seq.device)
        seq = seq + self.pos_emb(pos).unsqueeze(0)

        # Transformer
        out = self.transformer(seq)                                 # [B, N+1, D]
        cls_out = out[:, 0, :]                                      # [B, D]

        return {"mos": self.mlp(cls_out)}