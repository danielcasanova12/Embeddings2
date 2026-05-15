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
        embedding_columns: List[str],
        target_column: str = "mos",
        extra_columns: List[str] = None, # Colunas para Probing (Speaker, SNR, etc.)
        layer: int = -1,
        pool_time: bool = True,
    ):
        self.df = pd.read_csv(metadata_path)
        self.emb_cols    = embedding_columns
        self.target_col  = target_column
        self.extra_cols  = extra_columns or []
        self.layer       = layer
        self.pool_time   = pool_time

    def __len__(self):
        return len(self.df)

    def _load_emb(self, path: str) -> torch.Tensor:
        emb = torch.load(path, map_location="cpu", weights_only=True).float()
        if emb.dim() == 3:
            emb = emb[self.layer]
        
        if self.pool_time and emb.dim() == 2:
            emb = emb.mean(dim=0)
        return emb

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        
        embs = [self._load_emb(row[col]) for col in self.emb_cols]
        mos  = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        
        # Extra targets para Probing
        extras = {col: row[col] for col in self.extra_cols}
        
        return embs, mos, extras


def collate_fn(batch):
    """
    Agrupa lista de (embs, mos, extras).
    Se pool_time=False, faz padding nos embeddings que são sequências [T, D].
    """
    embs_list, mos_list, extras_list = zip(*batch)
    n_embs = len(embs_list[0])
    
    batched_embs = []
    masks = [] # máscaras de padding [B, T]

    for i in range(n_embs):
        samples = [s[i] for s in embs_list]
        if samples[0].dim() == 2: # Sequência [T, D]
            # Padding
            padded = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True)
            batched_embs.append(padded)
            
            # Gerar máscara
            B, T, D = padded.shape
            mask = torch.zeros(B, T, dtype=torch.bool)
            for b_idx, s in enumerate(samples):
                mask[b_idx, :s.shape[0]] = True
            masks.append(mask)
        else:
            # Já está poolado [D]
            batched_embs.append(torch.stack(samples))
            masks.append(None)
            
    mos = torch.stack(mos_list)
    return batched_embs, mos, masks, extras_list


def build_loaders(cfg: dict):
    ds_cfg = cfg["datasets"]
    tr_cfg = cfg["train"]
    emb_cols = [e["column"] for e in cfg["embeddings"]]
    
    # Detecção automática se precisamos de pool no dataset ou no modelo
    # Se qualquer embedding pedir ASP ou Stats, desativamos pool_time no Dataset
    model_pooling = cfg.get("model", {}).get("pooling", "mean")
    pool_time_dataset = (model_pooling.lower() == "mean") 
    
    # Probing extra columns
    extra_cols = cfg.get("probing", {}).get("extra_columns", [])

    def _ds(split_cfg):
        return MultiEmbeddingMOSDataset(
            metadata_path     = split_cfg["metadata_path"],
            embedding_columns = emb_cols,
            target_column     = split_cfg.get("target_column", "mos"),
            extra_columns     = extra_cols,
            pool_time         = pool_time_dataset
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