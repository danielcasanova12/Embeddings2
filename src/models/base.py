# src/models/base.py
"""
Lightning Module que envolve qualquer uma das 5 arquiteturas.
Gerencia treino, validação, teste, métricas e loss auxiliar de fatores.
"""

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from scipy.stats import pearsonr, spearmanr

from .architectures import (
    SingleEmbeddingMOS,
    CrossEmbeddingInteraction,
    ConcatFusionMOS,
    ReliabilityFusionMOS,
    TransformerFusionMOS,
    LinearProbeMOS,
)

ARCH_MAP = {
    "single":             SingleEmbeddingMOS,
    "cross_interaction":  CrossEmbeddingInteraction,
    "concat_fusion":      ConcatFusionMOS,
    "reliability_fusion": ReliabilityFusionMOS,
    "transformer_fusion": TransformerFusionMOS,
    "probing":            LinearProbeMOS,
}


class MOSPredictor(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        model_type = cfg.get("model_type", "concat_fusion")
        if model_type not in ARCH_MAP:
            raise ValueError(f"model_type '{model_type}' desconhecido. Opções: {list(ARCH_MAP)}")

        self.model = ARCH_MAP[model_type](cfg)
        self.mse   = nn.MSELoss()

        # Peso da loss auxiliar de fatores (usado apenas em reliability_fusion)
        rel_cfg = cfg.get("reliability", {})
        self.factor_weight = rel_cfg.get("factor_weight", 0.0)

        self._val_p,  self._val_t  = [], []
        self._test_p, self._test_t = [], []

    # ── forward ──────────────────────────────────────────────────

    def forward(self, embs, masks=None):
        return self.model(embs, masks)

    # ── loss ─────────────────────────────────────────────────────

    def _compute_loss(self, out: dict, target: torch.Tensor) -> torch.Tensor:
        # out["mos"] é a saída principal independente da task
        loss = self.mse(out["mos"], target)

        # Loss auxiliar de fatores (se architecture retornar "factors")
        if self.factor_weight > 0 and "factors" in out:
            factor_loss = torch.stack([
                self.mse(score, target) # Assume MSE para fatores
                for score in out["factors"].values()
            ]).mean()
            loss = loss + self.factor_weight * factor_loss

        return loss

    # ── steps ────────────────────────────────────────────────────

    def _get_target(self, mos, extras_list):
        if self.cfg.get("task") == "probing":
            target_col = self.cfg.get("probing", {}).get("target_column")
            # Extrai do extras_list
            targets = [e[target_col] for e in extras_list]
            return torch.tensor(targets, device=self.device).float()
        return mos

    def training_step(self, batch, _):
        embs, mos, masks, extras = batch
        target = self._get_target(mos, extras)
        out  = self(embs, masks)
        loss = self._compute_loss(out, target)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        embs, mos, masks, extras = batch
        target = self._get_target(mos, extras)
        out  = self(embs, masks)
        loss = self._compute_loss(out, target)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self._val_p.append(out["mos"].detach().cpu())
        self._val_t.append(target.detach().cpu())

    def on_validation_epoch_end(self):
        self._flush(self._val_p, self._val_t, "val")
        self._val_p.clear(); self._val_t.clear()

    def test_step(self, batch, _):
        embs, mos, masks, extras = batch
        target = self._get_target(mos, extras)
        out  = self(embs, masks)
        loss = self._compute_loss(out, target)
        self.log("test/loss", loss, on_epoch=True)
        self._test_p.append(out["mos"].detach().cpu())
        self._test_t.append(target.detach().cpu())

    def on_test_epoch_end(self):
        self._flush(self._test_p, self._test_t, "test")
        self._test_p.clear(); self._test_t.clear()

    def _flush(self, preds_list, targets_list, prefix):
        preds   = torch.cat(preds_list).numpy()
        targets = torch.cat(targets_list).numpy()
        mse      = float(((preds - targets) ** 2).mean())
        pearson  = float(pearsonr(preds, targets)[0])
        spearman = float(spearmanr(preds, targets)[0])
        self.log(f"{prefix}/mse",      mse,      on_epoch=True)
        self.log(f"{prefix}/pearson",  pearson,  on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/spearman", spearman, on_epoch=True, prog_bar=True)

    # ── optimizer ────────────────────────────────────────────────

    def configure_optimizers(self):
        opt_cfg = self.cfg["optimizer"]
        sch_cfg = self.cfg["scheduler"]
        tr_cfg  = self.cfg["trainer"]

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr           = opt_cfg["learning_rate"],
            weight_decay = opt_cfg["weight_decay"],
            eps          = opt_cfg["eps"],
            betas        = tuple(opt_cfg["betas"]),
        )

        name = sch_cfg["name"]
        if name == "cosine":
            warmup = sch_cfg.get("warmup_epochs", 0)
            total  = tr_cfg["max_epochs"]
            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / max(warmup, 1)
                progress = (epoch - warmup) / max(total - warmup, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))
            sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        elif name == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=10, factor=0.5)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": sched, "monitor": "val/spearman"}}
        return optimizer