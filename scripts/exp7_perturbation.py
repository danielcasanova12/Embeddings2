# scripts/exp7_perturbation.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

def calculate_sensitivity(emb_paths_quality: List[tuple], emb_paths_content: List[tuple]):
    """
    emb_paths_quality: lista de (path_audio1, path_audio2) com mesmo conteúdo, qualidades diferentes
    emb_paths_content: lista de (path_audio1, path_audio2) com frases diferentes, mesma qualidade
    """
    
    def get_dist(pairs):
        dists = []
        for p1, p2 in pairs:
            e1 = torch.load(p1, map_location="cpu").float()
            e2 = torch.load(p2, map_location="cpu").float()
            
            # Global mean pool se for sequência
            if e1.dim() > 1: e1 = e1.mean(dim=list(range(e1.dim()-1)))
            if e2.dim() > 1: e2 = e2.mean(dim=list(range(e2.dim()-1)))
            
            dist = 1 - F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
            dists.append(dist.item())
        return np.mean(dists)

    print("Calculando d_quality...")
    d_quality = get_dist(emb_paths_quality)
    
    print("Calculando d_content...")
    d_content = get_dist(emb_paths_content)
    
    ratio = d_content / (d_quality + 1e-9)
    return d_quality, d_content, ratio

if __name__ == "__main__":
    print("Script de Perturbação Semântica inicializado.")
    print("Para rodar este experimento, é necessário um CSV de pares conforme definido no prompt.")
    # Implementação simplificada: demonstração da lógica de cálculo de distância
