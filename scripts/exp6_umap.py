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

from src.dataset import build_loaders
from src.models import MOSPredictor

def run_umap(cfg_path, ckpt_path, output_dir="results/umap"):
    os.makedirs(output_dir, exist_ok=True)
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    
    # Adicionar configurações base se necessário (simular run_all.py)
    # Por simplicidade, assume-se que cfg já está completo ou que passamos o cfg_path do experimento
    
    model = MOSPredictor.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.eval().cuda()
    
    _, _, test_loader = build_loaders(cfg)
    
    latents = []
    mos_true = []
    
    print("Extraindo latentes...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            embs, mos, masks, _ = batch
            embs = [e.cuda() for e in embs]
            masks = [m.cuda() if m is not None else None for m in masks]
            
            z = model.model.extract_latent(embs, masks)
            latents.append(z.cpu().numpy())
            mos_true.append(mos.numpy())
            
    latents = np.concatenate(latents, axis=0)
    mos_true = np.concatenate(mos_true, axis=0)
    
    print(f"Rodando UMAP em {latents.shape}...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(latents)
    
    # Plot A: Colorido por MOS
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=mos_true, cmap='coolwarm', s=5, alpha=0.6)
    plt.colorbar(scatter, label='MOS Verdadeiro')
    plt.title(f"UMAP - Latent Space (Color by MOS)\n{os.path.basename(cfg_path)}")
    plt.savefig(os.path.join(output_dir, "umap_mos.png"))
    
    print(f"Resultados salvos em {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-k", "--checkpoint", required=True)
    args = parser.parse_args()
    run_umap(args.config, args.checkpoint)
