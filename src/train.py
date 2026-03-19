# src/train.py
"""
Executa um experimento completo (treino + teste).
Chamado por run_all.py — não é um script standalone.
"""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.dataset import build_loaders
from src.models import MOSPredictor


def run_experiment(cfg: dict) -> tuple[list[dict], dict]:
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    exp_name = cfg.get("experiment_name", "exp")

    train_loader, val_loader, test_loader = build_loaders(cfg)

    model    = MOSPredictor(cfg)
    ckpt_cfg = cfg["model_checkpoint"]
    ckpt_dir = os.path.join(ckpt_cfg["dirpath"], exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        monitor=ckpt_cfg["monitor"], mode=ckpt_cfg["mode"],
        save_last=ckpt_cfg["save_last"], dirpath=ckpt_dir,
        filename=ckpt_cfg["filename"], save_weights_only=True,
    )
    early_stop_cb = EarlyStopping(
        monitor=cfg["early_stopping"]["monitor"],
        mode=cfg["early_stopping"]["mode"],
        patience=cfg["early_stopping"]["patience"],
        verbose=False,
    )

    wb_cfg = cfg.get("wandb", {})
    logger = None
    if wb_cfg.get("enabled", False):
        logger = WandbLogger(
            project=wb_cfg.get("project", "mos-mlp"),
            entity=wb_cfg.get("entity", None),
            name=exp_name, config=cfg,
        )

    tr = cfg["trainer"]
    trainer = pl.Trainer(
        max_epochs=tr["max_epochs"],
        accelerator=tr["accelerator"],
        gradient_clip_val=tr["gradient_clip_val"],
        log_every_n_steps=tr["log_every_n_steps"],
        accumulate_grad_batches=tr["accumulate_grad_batches"],
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)
    test_results = trainer.test(model, test_loader, ckpt_path="best", verbose=False)
    test_metrics = test_results[0] if test_results else {}

    # Resumo por época (Lightning CSV logger)
    train_rows = _extract_epoch_metrics(trainer, exp_name)

    eval_row = {
        "experiment":        exp_name,
        "model_type":        cfg.get("model_type", "?"),
        "n_embeddings":      len(cfg.get("embeddings", [])),
        "embeddings":        "+".join(e["name"] for e in cfg.get("embeddings", [])),
        "best_val_spearman": float(checkpoint_cb.best_model_score or 0),
        "test_mse":          test_metrics.get("test/loss",     None),
        "test_pearson":      test_metrics.get("test/pearson",  None),
        "test_spearman":     test_metrics.get("test/spearman", None),
        "adapter_dim":       cfg["adapter"]["adapter_dim"],
        "hidden_dim":        cfg["mlp"]["hidden_dim"],
        "num_layers":        cfg["mlp"]["num_layers"],
        "learning_rate":     cfg["optimizer"]["learning_rate"],
        "batch_size":        cfg["train"]["batch_size"],
        "epochs_trained":    trainer.current_epoch,
    }

    if logger:
        import wandb; wandb.finish()

    return train_rows, eval_row


def _extract_epoch_metrics(trainer, exp_name: str) -> list[dict]:
    rows = []
    try:
        csv_logger = next(
            (l for l in (trainer.loggers or []) if hasattr(l, "experiment")), None
        )
        if csv_logger and hasattr(csv_logger.experiment, "metrics"):
            for m in csv_logger.experiment.metrics:
                if "epoch" in m:
                    rows.append({
                        "experiment":   exp_name,
                        "epoch":        int(m.get("epoch", 0)),
                        "train_loss":   m.get("train/loss_epoch", None),
                        "val_loss":     m.get("val/loss",         None),
                        "val_pearson":  m.get("val/pearson",      None),
                        "val_spearman": m.get("val/spearman",     None),
                    })
    except Exception:
        pass
    return rows