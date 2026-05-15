# src/models/architectures.py
"""
Implementação das 5 arquiteturas de predição de MOS.

Todas recebem como forward():
    embs: List[Tensor[B, D_i]]  — um tensor por embedding
    retornam: dict com pelo menos {"mos": Tensor[B]}
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .blocks import MLP, Adapter, build_adapters, get_pooling_layer


# ─────────────────────────────────────────────────────────────────────────────
# Base class to handle pooling
# ─────────────────────────────────────────────────────────────────────────────

class MultiEmbeddingBase(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.pooling_mode = cfg.get("model", {}).get("pooling", "mean")
        self.pooling_layers = nn.ModuleList([
            get_pooling_layer(self.pooling_mode, e["dim"])
            for e in cfg["embeddings"]
        ])

    def _pool_embs(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> List[torch.Tensor]:
        pooled = []
        for i, e in enumerate(embs):
            if e.dim() == 3: # [B, T, D]
                pooled.append(self.pooling_layers[i](e, masks[i]))
            else:
                pooled.append(e)
        return pooled


# ─────────────────────────────────────────────────────────────────────────────
# 1. Single Embedding
# ─────────────────────────────────────────────────────────────────────────────

class SingleEmbeddingMOS(MultiEmbeddingBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        emb_cfg = cfg["embeddings"][0]
        adp_cfg = cfg["adapter"]
        mlp_cfg = cfg["mlp"]
        
        # Ajuste de dimensão se pooling for Stats ou ASP (dobra a dimensão)
        in_dim = emb_cfg["dim"]
        if self.pooling_mode.lower() in ["stats", "asp"]:
            in_dim *= 2

        self.adapter = Adapter(
            dim_in      = in_dim,
            adapter_dim = adp_cfg["adapter_dim"],
            dropout     = adp_cfg.get("dropout", 0.1),
            use_norm    = adp_cfg.get("use_norm", True),
        )
        self.mlp = MLP(
            input_dim  = adp_cfg["adapter_dim"],
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg.get("output_dim", 1),
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
        )

    def forward(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> dict:
        pooled = self._pool_embs(embs, masks)
        x = self.adapter(pooled[0])
        return {"mos": self.mlp(x)}

    def extract_latent(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> torch.Tensor:
        pooled = self._pool_embs(embs, masks)
        return self.adapter(pooled[0])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cross-Embedding Interaction
# ─────────────────────────────────────────────────────────────────────────────

class CrossEmbeddingInteraction(MultiEmbeddingBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        emb_cfgs = cfg["embeddings"]
        adp_cfg  = cfg["adapter"]
        mlp_cfg  = cfg["mlp"]
        D        = adp_cfg["adapter_dim"]
        
        # Atualiza dims dos adapters baseado no pooling
        modified_emb_cfgs = []
        for e in emb_cfgs:
            d = e["dim"]
            if self.pooling_mode.lower() in ["stats", "asp"]:
                d *= 2
            modified_emb_cfgs.append({"dim": d, "name": e["name"]})

        self.adapters = build_adapters(modified_emb_cfgs, adp_cfg)
        
        # Exp 3 pede interação Hadamard. 
        # Se mode for 'simple_interaction', usa apenas o cross-product.
        # Caso contrário, concatena originais + cross-product.
        self.interaction_mode = cfg.get("interaction_mode", "full")
        N = len(emb_cfgs)
        n_pairs = N * (N - 1) // 2
        
        if self.interaction_mode == "hadamard_only":
            input_dim = D # assume N=2 para o Experimento 3
        else:
            input_dim = D * N + D * n_pairs

        self.mlp = MLP(
            input_dim  = input_dim,
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg.get("output_dim", 1),
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
        )

    def forward(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> dict:
        x = self.extract_latent(embs, masks)
        return {"mos": self.mlp(x)}

    def extract_latent(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> torch.Tensor:
        pooled = self._pool_embs(embs, masks)
        adapted = [adp(e) for adp, e in zip(self.adapters, pooled)]
        
        if self.interaction_mode == "hadamard_only" and len(adapted) == 2:
            x = adapted[0] * adapted[1]
        else:
            pairs = [
                adapted[i] * adapted[j]
                for i in range(len(adapted))
                for j in range(i + 1, len(adapted))
            ]
            x = torch.cat(adapted + pairs, dim=-1)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-Embedding Concat Fusion
# ─────────────────────────────────────────────────────────────────────────────

class ConcatFusionMOS(MultiEmbeddingBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        emb_cfgs = cfg["embeddings"]
        adp_cfg  = cfg["adapter"]
        mlp_cfg  = cfg["mlp"]
        D        = adp_cfg["adapter_dim"]

        modified_emb_cfgs = []
        for e in emb_cfgs:
            d = e["dim"]
            if self.pooling_mode.lower() in ["stats", "asp"]:
                d *= 2
            modified_emb_cfgs.append({"dim": d, "name": e["name"]})

        self.adapters = build_adapters(modified_emb_cfgs, adp_cfg)
        self.mlp = MLP(
            input_dim  = D * len(emb_cfgs),
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg.get("output_dim", 1),
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
        )

    def forward(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> dict:
        x = self.extract_latent(embs, masks)
        return {"mos": self.mlp(x)}

    def extract_latent(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> torch.Tensor:
        pooled = self._pool_embs(embs, masks)
        adapted = [adp(e) for adp, e in zip(self.adapters, pooled)]
        return torch.cat(adapted, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Linear Probing (Exp 5)
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbeMOS(MultiEmbeddingBase):
    """
    Linear Probe: Embedding (pooled) -> Linear Layer -> Output
    """
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        emb_cfg = cfg["embeddings"][0] # Probing é single-embedding no Exp 5
        
        in_dim = emb_cfg["dim"]
        if self.pooling_mode.lower() in ["stats", "asp"]:
            in_dim *= 2
            
        self.linear = nn.Linear(in_dim, cfg.get("output_dim", 1))

    def forward(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> dict:
        pooled = self._pool_embs(embs, masks)
        return {"mos": self.linear(pooled[0])}


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