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

    if not cfg.get("evaluate_only", False):
        trainer.fit(model, train_loader, val_loader)
        ckpt_path = "best"
    else:
        print(f"Modo EVALUATE_ONLY ativado. Carregando: {cfg.get('checkpoint_path')}")
        ckpt_path = cfg.get("checkpoint_path")

    test_results = trainer.test(model, test_loader, ckpt_path=ckpt_path, verbose=False)
    test_metrics = test_results[0] if test_results else {}

    # Resumo por época (Lightning CSV logger)
    train_rows = _extract_epoch_metrics(trainer, exp_name) if not cfg.get("evaluate_only", False) else []

    eval_row = {
        "experiment":        exp_name,
        "checkpoint_path":   checkpoint_cb.best_model_path,
        "model_type":        cfg.get("model_type", "?"),
        "n_embeddings":      len(cfg.get("embeddings", [])),
        "embeddings":        "+".join(e["name"] for e in cfg.get("embeddings", [])),
        "best_val_spearman": float(checkpoint_cb.best_model_score or 0),
        "test_pearson":      test_metrics.get("test/pearson",  None),
        "test_spearman":     test_metrics.get("test/spearman", None),
        "adapter_dim":       cfg["adapter"]["adapter_dim"],
        "hidden_dim":        cfg["mlp"]["hidden_dim"],
        "num_layers":        cfg["mlp"]["num_layers"],
        "learning_rate":     cfg["optimizer"]["learning_rate"],
        "batch_size":        cfg["train"]["batch_size"],
        "epochs_trained":    trainer.current_epoch,
    }

    # --- Extração de Detalhes Adicionais para o PDF ---
    if cfg["model_type"] == "weighted_fusion":
        # Extrai pesos médios do loader de teste
        model.eval()
        all_w = []
        with torch.no_grad():
            for batch in test_loader:
                embs, mos, masks, _ = batch
                embs = [e.to(model.device) for e in embs]
                masks = [m.to(model.device) if m is not None else None for m in masks]
                out = model(embs, masks)
                if "weights" in out:
                    all_w.append(out["weights"].cpu())
        if all_w:
            avg_w = torch.cat(all_w).mean(dim=0).tolist()
            weights_dict = {cfg["embeddings"][i]["name"]: avg_w[i] for i in range(len(avg_w))}
            eval_row["weights"] = weights_dict
            eval_row["Detalhes Extra"] = "Pesos extraídos"

    elif cfg["model_type"] == "mutual_info_reg":
        # Extrai CKA médio do loader de teste
        model.eval()
        all_cka = []
        from src.models.blocks import linear_cka
        with torch.no_grad():
            for batch in test_loader:
                embs, mos, masks, _ = batch
                embs = [e.to(model.device) for e in embs]
                masks = [m.to(model.device) if m is not None else None for m in masks]
                out = model(embs, masks)
                if "latents" in out and len(out["latents"]) >= 2:
                    latents = out["latents"]
                    c_val = 0; c_count = 0
                    for i in range(len(latents)):
                        for j in range(i+1, len(latents)):
                            c_val += linear_cka(latents[i], latents[j]).item()
                            c_count += 1
                    all_cka.append(c_val / c_count)
        if all_cka:
            avg_cka = sum(all_cka) / len(all_cka)
            eval_row["Detalhes Extra"] = f"Avg CKA: {avg_cka:.4f}"

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