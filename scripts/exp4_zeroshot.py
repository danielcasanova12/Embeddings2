# scripts/exp4_zeroshot.py
import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, ttest_rel
from sklearn.metrics import mean_squared_error
from src.dataset import build_loaders
from src.models import MOSPredictor

def run_zeroshot(cfg_paths: list, ckpt_full: str, ckpt_minus_c: str, output_dir: str = "results/zeroshot"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Carrega os modelos
    # Nota: assume que a config base para carregar o checkpoint está no primeiro dataset
    base_cfg = OmegaConf.to_container(OmegaConf.load(cfg_paths[0]), resolve=True)
    
    model_full = MOSPredictor.load_from_checkpoint(ckpt_full, cfg=base_cfg).eval().cuda()
    model_minc = MOSPredictor.load_from_checkpoint(ckpt_minus_c, cfg=base_cfg).eval().cuda()
    
    results = []
    
    for cfg_path in cfg_paths:
        cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        ds_name = cfg.get("datasets", {}).get("name", os.path.basename(cfg_path))
        print(f"\nAvaliando Zero-Shot no dataset: {ds_name}")
        
        _, _, test_loader = build_loaders(cfg)
        
        preds_full, preds_minc, targets = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                embs, mos, masks, _ = batch
                embs = [e.cuda() for e in embs]
                masks = [m.cuda() if m is not None else None for m in masks]
                
                # Predição FULL
                out_f = model_full(embs, masks)
                preds_full.append(out_f["mos"].cpu().numpy())
                
                # Predição -C (precisa garantir que a lista de embeddings no batch bate com o que o modelo espera)
                # O batch vem com TODOS os embeddings da config do dataset. 
                # Se o model_minc foi treinado com menos embeddings, ele vai ignorar os extras ou 
                # precisamos filtrar aqui se as arquiteturas forem rígidas.
                # No nosso MOSPredictor, ele usa cfg['embeddings'] da sua própria criação.
                out_mc = model_minc(embs, masks)
                preds_minc.append(out_mc["mos"].cpu().numpy())
                
                targets.append(mos.numpy())
                
        preds_full = np.concatenate(preds_full)
        preds_minc = np.concatenate(preds_minc)
        targets    = np.concatenate(targets)
        
        # 1. Métricas Básicas
        def get_metrics(p, t):
            return {
                "pearson":  pearsonr(p, t)[0],
                "spearman": spearmanr(p, t)[0],
                "rmse":     np.sqrt(mean_squared_error(p, t))
            }
            
        m_full = get_metrics(preds_full, targets)
        m_minc = get_metrics(preds_minc, targets)
        
        # 2. Bootstrap (1000 iterações) para Pearson
        n_boot = 1000
        boot_full, boot_minc = [], []
        for _ in range(n_boot):
            idx = np.random.choice(len(targets), len(targets), replace=True)
            boot_full.append(pearsonr(preds_full[idx], targets[idx])[0])
            boot_minc.append(pearsonr(preds_minc[idx], targets[idx])[0])
            
        # 3. Paired t-test (comparando o erro absoluto por amostra ou a predição direta)
        # README sugere comparar as predições/erros.
        err_full = np.abs(preds_full - targets)
        err_minc = np.abs(preds_minc - targets)
        t_stat, p_val = ttest_rel(err_full, err_minc)
        
        results.append({
            "Dataset": ds_name,
            "FULL_Pearson": f"{m_full['pearson']:.4f} ± {np.std(boot_full):.4f}",
            "FULL_RMSE": f"{m_full['rmse']:.4f}",
            "-C_Pearson": f"{m_minc['pearson']:.4f} ± {np.std(boot_minc):.4f}",
            "-C_RMSE": f"{m_minc['rmse']:.4f}",
            "t-test_p": f"{p_val:.4e}",
            "Better": "FULL" if m_full['pearson'] > m_minc['pearson'] else "-C"
        })
        
    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, "zeroshot_results.csv")
    results_df.to_csv(output_path, index=False)
    print("\n--- Resultados Zero-Shot Generalization ---")
    print(results_df.to_string(index=False))
    print(f"\nSalvo em: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs", nargs='+', required=True, help="Configs dos datasets alvo (NISQA, SOMOS...)")
    parser.add_argument("--full", required=True, help="Checkpoint do modelo FULL")
    parser.add_argument("--minc", required=True, help="Checkpoint do modelo -C")
    parser.add_argument("-o", "--output", default="results/zeroshot")
    args = parser.parse_args()
    run_zeroshot(args.configs, args.full, args.minc, args.output)
