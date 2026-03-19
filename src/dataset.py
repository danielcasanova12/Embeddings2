# src/dataset.py
"""
Dataset que carrega N embeddings .pt por amostra + label MOS.

CSV esperado (uma linha por áudio):
    whisper_path, contentvec_path, speaker_path, f0_path, mos
    /data/w/f1.pt, /data/cv/f1.pt, /data/spk/f1.pt, /data/f0/f1.pt, 4.2
    ...

Shapes suportados por embedding:
    [D]       → usado diretamente
    [T, D]    → mean-pool em T → [D]
    [L, T, D] → seleciona layer=-1, mean-pool → [D]
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List


class MultiEmbeddingMOSDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        embedding_columns: List[str],     # lista de colunas com caminhos de .pt
        target_column: str = "mos",
        layer: int = -1,
        pool_time: bool = True,
    ):
        self.df = pd.read_csv(metadata_path)
        self.emb_cols   = embedding_columns
        self.target_col = target_column
        self.layer      = layer
        self.pool_time  = pool_time

    def __len__(self):
        return len(self.df)

    def _load_emb(self, path: str) -> torch.Tensor:
        emb = torch.load(path, map_location="cpu", weights_only=True).float()
        if emb.dim() == 3:
            emb = emb[self.layer]
        if emb.dim() == 2 and self.pool_time:
            emb = emb.mean(dim=0)
        return emb

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        embs = [self._load_emb(row[col]) for col in self.emb_cols]
        mos  = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return embs, mos


def collate_fn(batch):
    """Agrupa lista de (embs, mos) em tensores por embedding."""
    embs_list, mos_list = zip(*batch)
    n_embs = len(embs_list[0])
    # Para cada posição i, empilha todos os samples → [B, D]
    batched_embs = [
        torch.stack([sample[i] for sample in embs_list])
        for i in range(n_embs)
    ]
    mos = torch.stack(mos_list)
    return batched_embs, mos


def build_loaders(cfg: dict):
    ds_cfg = cfg["datasets"]
    tr_cfg = cfg["train"]
    emb_cols = [e["column"] for e in cfg["embeddings"]]

    def _ds(split_cfg):
        return MultiEmbeddingMOSDataset(
            metadata_path     = split_cfg["metadata_path"],
            embedding_columns = emb_cols,
            target_column     = split_cfg["target_column"],
        )

    def _loader(ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size  = tr_cfg["batch_size"],
            shuffle     = shuffle,
            num_workers = tr_cfg["num_workers"],
            pin_memory  = True,
            collate_fn  = collate_fn,
        )

    train_ds = _ds(ds_cfg["train"])
    val_ds   = _ds(ds_cfg["val"])
    test_ds  = _ds(ds_cfg["test"])

    return (
        _loader(train_ds, shuffle=tr_cfg["shuffle"]),
        _loader(val_ds),
        _loader(test_ds),
    )