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
from pathlib import Path


EMBEDDING_COLUMN_MAP = {
    "whisper": "whisper_path",
    "contentvec": "contentvec_path",
    "speaker": "speaker_path",
    "f0": "f0_path",
    "hubert": "hubert_path",
    "wavlm": "wavlm_path",
    "wav2vec2": "wav2vec2_path",
}

EMBEDDING_DIM_MAP = {
    "whisper": 1280,
    "contentvec": 768,
    "speaker": 192,
    "f0": 1,
    "hubert": 768,
    "wavlm": 768,
    "wav2vec2": 768,
}


def _resolve_case_insensitive_path(path_str: str) -> str:
    path = Path(path_str)
    if path.exists():
        return str(path)

    base = Path(path.anchor) if path.is_absolute() else Path(".")
    parts = [p for p in path.parts if p not in ("", path.anchor)]
    current = base

    for part in parts:
        if not current.exists():
            return path_str

        exact = current / part
        if exact.exists():
            current = exact
            continue

        lowered = part.lower()
        matches = [child for child in current.iterdir() if child.name.lower() == lowered]
        if len(matches) != 1:
            return path_str
        current = matches[0]

    return str(current)


def _embedding_column(emb_cfg: dict) -> str:
    column = emb_cfg.get("column")
    if column:
        return column

    name = emb_cfg.get("name", "").lower()
    if name in EMBEDDING_COLUMN_MAP:
        return EMBEDDING_COLUMN_MAP[name]

    raise KeyError(
        f"Embedding '{emb_cfg.get('name', '?')}' sem campo 'column' e sem mapeamento conhecido."
    )


def normalize_embedding_configs(emb_cfgs: List[dict]) -> List[dict]:
    normalized = []
    for emb_cfg in emb_cfgs:
        item = dict(emb_cfg)
        name = item.get("name", "").lower()

        if not item.get("column") and name in EMBEDDING_COLUMN_MAP:
            item["column"] = EMBEDDING_COLUMN_MAP[name]

        if name in EMBEDDING_DIM_MAP:
            item["dim"] = EMBEDDING_DIM_MAP[name]

        normalized.append(item)

    return normalized


class MultiEmbeddingMOSDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        embedding_specs: List[dict],
        target_column: str = "mos",
        extra_columns: List[str] = None, # Colunas para Probing (Speaker, SNR, etc.)
        layer: int = -1,
        pool_time: bool = True,
    ):
        self.df = pd.read_csv(metadata_path)
        self.emb_specs   = embedding_specs
        self.emb_cols    = [spec["column"] for spec in embedding_specs]
        self.target_col  = target_column
        self.extra_cols  = extra_columns or []
        self.layer       = layer
        self.pool_time   = pool_time

    def __len__(self):
        return len(self.df)

    def _load_emb(self, path: str, emb_spec: dict) -> torch.Tensor:
        emb = torch.load(path, map_location="cpu", weights_only=True).float()
        expected_dim = emb_spec.get("dim")
        if emb.dim() == 3:
            emb = emb[self.layer]

        if emb.dim() == 1:
            if expected_dim is not None and emb.shape[0] == expected_dim:
                return emb

            # Sequência escalar 1D, como F0 frame a frame.
            if self.pool_time:
                emb = emb.mean(dim=0, keepdim=True)
            else:
                emb = emb.unsqueeze(-1)
            return emb

        if self.pool_time and emb.dim() == 2:
            emb = emb.mean(dim=0)
        return emb

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        
        embs = [
            self._load_emb(row[col], spec)
            for col, spec in zip(self.emb_cols, self.emb_specs)
        ]
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
    embedding_specs = normalize_embedding_configs(cfg["embeddings"])
    
    # Detecção automática se precisamos de pool no dataset ou no modelo
    # Se qualquer embedding pedir ASP ou Stats, desativamos pool_time no Dataset
    model_pooling = cfg.get("model", {}).get("pooling", "mean")
    pool_time_dataset = (model_pooling.lower() == "mean") 
    
    # Probing extra columns
    extra_cols = cfg.get("probing", {}).get("extra_columns", [])

    def _ds(split_cfg):
        return MultiEmbeddingMOSDataset(
            metadata_path     = _resolve_case_insensitive_path(split_cfg["metadata_path"]),
            embedding_specs   = embedding_specs,
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
