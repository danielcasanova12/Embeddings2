# scripts/exp7_perturbation.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from typing import List, Dict
from tqdm import tqdm

def calculate_sensitivity(csv_path: str, output_dir: str = "results/perturbation"):
    """
    csv_path: CSV com colunas:
        - pair_type: 'quality' ou 'content'
        - embedding_type: whisper, speaker, etc.
        - path_q1, path_q2 (se quality)
        - path_c1, path_c2 (se content)
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    results = []
    unique_embs = df['embedding_type'].unique()
    
    # Helper to load and pool
    def get_emb(path):
        if pd.isna(path): return None
        e = torch.load(path, map_location="cpu", weights_only=True).float()
        if e.dim() == 3: e = e[-1]
        if e.dim() == 2: e = e.mean(dim=0)
        return e.unsqueeze(0)

    for emb_type in unique_embs:
        print(f"Processando embedding: {emb_type}")
        sub_df = df[df['embedding_type'] == emb_type]
        
        d_quality_list = []
        d_content_list = []
        
        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
            if row['pair_type'] == 'quality':
                e1 = get_emb(row['path_q1'])
                e2 = get_emb(row['path_q2'])
                if e1 is not None and e2 is not None:
                    dist = 1 - F.cosine_similarity(e1, e2).item()
                    d_quality_list.append(dist)
            else:
                e1 = get_emb(row['path_c1'])
                e2 = get_emb(row['path_c2'])
                if e1 is not None and e2 is not None:
                    dist = 1 - F.cosine_similarity(e1, e2).item()
                    d_content_list.append(dist)
            
        if not d_quality_list or not d_content_list:
            print(f"Aviso: Dados insuficientes para {emb_type}. Q:{len(d_quality_list)}, C:{len(d_content_list)}")
            continue

        d_q_mean, d_q_std = np.mean(d_quality_list), np.std(d_quality_list)
        d_c_mean, d_c_std = np.mean(d_content_list), np.std(d_content_list)
        
        # Como as listas podem ter tamanhos diferentes, o ratio é calculado sobre as médias
        ratio = d_c_mean / (d_q_mean + 1e-9)
        
        results.append({
            "Embedding": emb_type,
            "d_quality (mean)": d_q_mean,
            "d_quality (std)": d_q_std,
            "d_content (mean)": d_c_mean,
            "d_content (std)": d_c_std,
            "Ratio (C/Q)": ratio,
            "N_quality": len(d_quality_list),
            "N_content": len(d_content_list)
        })
        
    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, "perturbation_results.csv")
    results_df.to_csv(output_path, index=False)
    
    print("\n--- Tabela Consolidada de Perturbação ---")
    print(results_df.to_string(index=False))
    print(f"\nResultados salvos em {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_csv", required=True, help="CSV com pares de áudio e tipos de embedding")
    parser.add_argument("-o", "--output_dir", default="results/perturbation")
    args = parser.parse_args()
    calculate_sensitivity(args.input_csv, args.output_dir)
