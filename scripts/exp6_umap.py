# scripts/exp6_umap.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from omegaconf import OmegaConf
from tqdm import tqdm
from typing import List

from src.dataset import build_loaders
from src.models import MOSPredictor

def run_umap(cfg_paths: List[str], ckpt_path: str, output_dir: str = "results/umap"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Carrega o modelo de um checkpoint (assume que a config base está no checkpoint ou passada)
    # Para o UMAP, geralmente usamos o melhor modelo (ex: -C ou W+S)
    first_cfg = OmegaConf.to_container(OmegaConf.load(cfg_paths[0]), resolve=True)
    model = MOSPredictor.load_from_checkpoint(ckpt_path, cfg=first_cfg)
    model.eval().cuda()
    
    all_latents = []
    all_mos_true = []
    all_datasets = []
    
    for cfg_path in cfg_paths:
        cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        ds_name = cfg.get("datasets", {}).get("name", os.path.basename(cfg_path))
        print(f"Extraindo latentes para o dataset: {ds_name}...")
        
        _, _, test_loader = build_loaders(cfg)
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                embs, mos, masks, _ = batch
                embs = [e.cuda() for e in embs]
                masks = [m.cuda() if m is not None else None for m in masks]
                
                # Extrai o vetor latente ANTES do regressor final (MLP)
                # O método extract_latent já está implementado em architectures.py
                z = model.model.extract_latent(embs, masks)
                
                all_latents.append(z.cpu().numpy())
                all_mos_true.append(mos.numpy())
                all_datasets.extend([ds_name] * len(mos))
                
    latents = np.concatenate(all_latents, axis=0)
    mos_true = np.concatenate(all_mos_true, axis=0)
    
    print(f"Rodando UMAP em {latents.shape} (neighbors=15, dist=0.1, metric=cosine)...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(latents)
    
    # Plot A: Colorido por MOS
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=mos_true, cmap='coolwarm', s=5, alpha=0.6)
    plt.colorbar(scatter, label='MOS Verdadeiro')
    plt.title(f"UMAP - Perceptual Quality Space (Color by MOS)")
    plt.savefig(os.path.join(output_dir, "umap_mos.png"), dpi=300)
    plt.close()
    
    # Plot B: Colorido por Dataset
    plt.figure(figsize=(12, 10))
    unique_ds = list(set(all_datasets))
    colors = sns.color_palette("husl", len(unique_ds))
    for i, ds in enumerate(unique_ds):
        idx = [j for j, d in enumerate(all_datasets) if d == ds]
        plt.scatter(embedding[idx, 0], embedding[idx, 1], label=ds, s=5, alpha=0.6, color=colors[i])
    
    plt.legend(markerscale=3)
    plt.title(f"UMAP - Corpus Bias Analysis (Color by Dataset)")
    plt.savefig(os.path.join(output_dir, "umap_datasets.png"), dpi=300)
    plt.close()
    
    print(f"Resultados (Plots A e B) salvos em {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", nargs='+', required=True, help="Lista de caminhos para configs de datasets (yaml)")
    parser.add_argument("-k", "--checkpoint", required=True, help="Caminho do checkpoint do modelo treinado")
    parser.add_argument("-o", "--output", default="results/umap")
    args = parser.parse_args()
    run_umap(args.configs, args.checkpoint, args.output)
