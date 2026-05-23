# src/models/architectures.py
"""
Implementação das 8 arquiteturas de predição de MOS.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .blocks import MLP, Adapter, build_adapters, get_pooling_layer, linear_cka, GRL


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

    def extract_latent(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> torch.Tensor:
        """Default fallback: concatenation of pooled embeddings."""
        pooled = self._pool_embs(embs, masks)
        return torch.cat(pooled, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Single Embedding
# ─────────────────────────────────────────────────────────────────────────────

class SingleEmbeddingMOS(MultiEmbeddingBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        emb_cfg = cfg["embeddings"][0]
        adp_cfg = cfg["adapter"]
        mlp_cfg = cfg["mlp"]
        
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
            sigmoid_scale = cfg.get("sigmoid_scale", False),
        )

    def forward(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> dict:
        x = self.extract_latent(embs, masks)
        return {"mos": self.mlp(x)}

    def extract_latent(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> torch.Tensor:
        pooled = self._pool_embs(embs, masks)
        return self.adapter(pooled[0])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cross-Embedding Interaction (Exp 3)
# ─────────────────────────────────────────────────────────────────────────────

class CrossEmbeddingInteraction(MultiEmbeddingBase):
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
        self.interaction_mode = cfg.get("interaction_mode", "full")
        N = len(emb_cfgs)
        n_pairs = N * (N - 1) // 2
        
        if self.interaction_mode == "hadamard_only":
            input_dim = D 
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
            sigmoid_scale = cfg.get("sigmoid_scale", False),
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
# 3. Multi-Embedding Concat Fusion (Exp 2)
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
            sigmoid_scale = cfg.get("sigmoid_scale", False),
        )

    def forward(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> dict:
        x = self.extract_latent(embs, masks)
        return {"mos": self.mlp(x)}

    def extract_latent(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> torch.Tensor:
        pooled = self._pool_embs(embs, masks)
        adapted = [adp(e) for adp, e in zip(self.adapters, pooled)]
        return torch.cat(adapted, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Reliability Fusion + Perceptual Factors (Exp 4 in prompt.md)
# ─────────────────────────────────────────────────────────────────────────────

class ReliabilityFusionMOS(MultiEmbeddingBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        emb_cfgs     = cfg["embeddings"]
        adp_cfg      = cfg["adapter"]
        mlp_cfg      = cfg["mlp"]
        rel_cfg      = cfg.get("reliability", {})
        D            = adp_cfg["adapter_dim"]
        factor_names = rel_cfg.get("factor_names", [])

        self.adapters = build_adapters(emb_cfgs, adp_cfg)

        self.reliability_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(D, 1), nn.Sigmoid())
            for _ in emb_cfgs
        ])

        self.mos_mlp = MLP(
            input_dim  = D,
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg.get("output_dim", 1),
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
            sigmoid_scale = cfg.get("sigmoid_scale", False),
        )

        self.factor_names = factor_names
        self.factor_heads = nn.ModuleList([
            nn.Linear(D, 1) for _ in factor_names
        ])

    def forward(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> dict:
        pooled = self._pool_embs(embs, masks)
        adapted = [adp(e) for adp, e in zip(self.adapters, pooled)]

        weights = torch.cat([rh(a) for rh, a in zip(self.reliability_heads, adapted)], dim=-1)
        weights = torch.softmax(weights, dim=-1)

        stacked = torch.stack(adapted, dim=1)
        fused   = (stacked * weights.unsqueeze(-1)).sum(dim=1)

        mos = self.mos_mlp(fused)
        factor_scores = {name: self.factor_heads[i](fused).squeeze(-1) for i, name in enumerate(self.factor_names)}

        return {"mos": mos, "factors": factor_scores, "weights": weights}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Transformer Fusion (Exp 5 in prompt.md)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerFusionMOS(MultiEmbeddingBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        emb_cfgs = cfg["embeddings"]
        adp_cfg  = cfg["adapter"]
        mlp_cfg  = cfg["mlp"]
        tr_cfg   = cfg.get("transformer", {})
        D        = adp_cfg["adapter_dim"]
        N        = len(emb_cfgs)

        self.adapters = build_adapters(emb_cfgs, adp_cfg)
        self.cls_token = nn.Parameter(torch.randn(1, 1, D))
        self.pos_emb = nn.Embedding(N + 1, D)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=tr_cfg.get("num_heads", 4), dim_feedforward=tr_cfg.get("feedforward_dim", 512),
            dropout=tr_cfg.get("dropout", 0.1), batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tr_cfg.get("num_layers", 2))

        self.mlp = MLP(
            input_dim  = D,
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg.get("output_dim", 1),
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
            sigmoid_scale = cfg.get("sigmoid_scale", False),
        )

    def forward(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> dict:
        pooled = self._pool_embs(embs, masks)
        B = pooled[0].shape[0]
        adapted = [adp(e) for adp, e in zip(self.adapters, pooled)]
        tokens = torch.stack(adapted, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        pos = torch.arange(seq.shape[1], device=seq.device)
        seq = seq + self.pos_emb(pos).unsqueeze(0)
        out = self.transformer(seq)
        cls_out = out[:, 0, :]
        return {"mos": self.mlp(cls_out)}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Weighted Sum Fusion (Exp 9 - prompt2.md)
# ─────────────────────────────────────────────────────────────────────────────

class WeightedSumFusionMOS(MultiEmbeddingBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        emb_cfgs = cfg["embeddings"]
        adp_cfg  = cfg["adapter"]
        mlp_cfg  = cfg["mlp"]
        D        = adp_cfg["adapter_dim"]

        self.adapters = build_adapters(emb_cfgs, adp_cfg)
        self.weight_mlp = MLP(
            input_dim=D * len(emb_cfgs), hidden_dim=128, num_layers=1, output_dim=len(emb_cfgs),
            dropout=0.1, input_norm=True
        )
        self.mlp = MLP(
            input_dim  = D,
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg.get("output_dim", 1),
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
            sigmoid_scale = cfg.get("sigmoid_scale", False),
        )

    def forward(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> dict:
        pooled = self._pool_embs(embs, masks)
        adapted = [adp(e) for adp, e in zip(self.adapters, pooled)]
        z_concat = torch.cat(adapted, dim=-1)
        weights = torch.softmax(self.weight_mlp(z_concat), dim=-1)
        stacked = torch.stack(adapted, dim=1)
        f_weighted = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return {"mos": self.mlp(f_weighted), "weights": weights}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Mutual Info Regularized Fusion (Exp 10 - prompt2.md)
# ─────────────────────────────────────────────────────────────────────────────

class RegularizedMutualInfoMOS(MultiEmbeddingBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        emb_cfgs = cfg["embeddings"]
        adp_cfg  = cfg["adapter"]
        mlp_cfg  = cfg["mlp"]
        D        = adp_cfg["adapter_dim"]

        self.adapters = build_adapters(emb_cfgs, adp_cfg)
        self.mlp = MLP(
            input_dim  = D * len(emb_cfgs),
            hidden_dim = mlp_cfg["hidden_dim"],
            num_layers = mlp_cfg["num_layers"],
            output_dim = cfg.get("output_dim", 1),
            dropout    = mlp_cfg["dropout"],
            activation = mlp_cfg["activation"],
            input_norm = mlp_cfg.get("input_norm", True),
            sigmoid_scale = cfg.get("sigmoid_scale", False),
        )
        self.use_grl = cfg.get("use_grl", False)
        if self.use_grl:
            self.grl = GRL(alpha=cfg.get("grl_lambda", 1.0))
            # Adversarial Predictor: tries to predict reference (emb[0]) from others
            self.adv_predictor = nn.Linear(D, D)

    def forward(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> dict:
        pooled = self._pool_embs(embs, masks)
        adapted = [adp(e) for adp, e in zip(self.adapters, pooled)]
        z_concat = torch.cat(adapted, dim=-1)
        out = {"mos": self.mlp(z_concat), "latents": adapted}
        if self.use_grl and len(adapted) >= 2:
            # Variant B: adversarial redundancy removal
            # Predict reference (adapted[0]) from second embedding (adapted[1])
            z_ref, z_other = adapted[0], adapted[1]
            pred = self.adv_predictor(self.grl(z_other))
            out["adv_pred"] = pred
            out["adv_target"] = z_ref
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. Linear Probing (Exp 5 in protocol)
# ─────────────────────────────────────────────────────────────────────────────

class LinearProbeMOS(MultiEmbeddingBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        emb_cfg = cfg["embeddings"][0]
        in_dim = emb_cfg["dim"]
        if self.pooling_mode.lower() in ["stats", "asp"]:
            in_dim *= 2
        self.linear = nn.Linear(in_dim, cfg.get("output_dim", 1))

    def forward(self, embs: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> dict:
        pooled = self._pool_embs(embs, masks)
        return {"mos": self.linear(pooled[0])}
