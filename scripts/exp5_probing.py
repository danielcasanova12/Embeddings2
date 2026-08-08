# scripts/exp5_probing.py
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, r2_score, f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from src.dataset import build_loaders, normalize_embedding_configs

class LinearProbe(nn.Module):
    """Strictly linear probe: Input -> Linear -> Output"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def train_probe(model, loader, val_loader, task_type="regression", epochs=20, lr=1e-3, target_col=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()
    model.cuda()
    
    best_val_metric = -float('inf') if task_type != "mse" else float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            embs, mos, masks, extras = batch
            x = embs[0].cuda() 
            if x.dim() == 3: x = x.mean(dim=1)
            
            if task_type == "regression":
                if target_col and target_col in extras[0]:
                    y = torch.tensor([e[target_col] for e in extras]).float().cuda()
                else:
                    y = mos.cuda()
                out = model(x).squeeze(-1)
            elif task_type == "semantic_regression":
                y = torch.tensor([e[target_col] for e in extras]).float().cuda()
                out = model(x)
            else:
                y = torch.tensor([e[target_col] for e in extras]).long().cuda()
                out = model(x)
                
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                embs, mos, masks, extras = batch
                x = embs[0].cuda()
                if x.dim() == 3: x = x.mean(dim=1)
                
                if task_type == "regression":
                    y = mos if target_col == "mos" else torch.tensor([e[target_col] for e in extras]).float()
                    out = model(x).squeeze(-1).cpu()
                elif task_type == "semantic_regression":
                    y = torch.tensor([e[target_col] for e in extras]).float()
                    out = model(x).cpu()
                else:
                    y = torch.tensor([e[target_col] for e in extras]).long()
                    out = model(x).argmax(dim=-1).cpu()
                
                all_preds.append(out.numpy())
                all_targets.append(y.numpy())
                    
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        if task_type == "regression":
            metric = pearsonr(all_preds, all_targets)[0]
        elif task_type == "semantic_regression":
            # Usa cosine similarity média para semântica
            cos_sims = [cosine_similarity(p, t) for p, t in zip(all_preds, all_targets)]
            metric = np.mean(cos_sims)
        else:
            metric = accuracy_score(all_targets, all_preds)
            
        # Early stopping logic (best metric)
        if (task_type != "mse" and metric > best_val_metric) or (task_type == "mse" and metric < best_val_metric):
            best_val_metric = metric
            best_state = model.state_dict()
            
    model.load_state_dict(best_state)
    return best_val_metric

def run_probing_experiment(
    cfg_path: str,
    output_dir: str = "results/probing",
    dataset_config: str = "configs/datasets/bvcc.yaml",
    model_defaults: str = "configs/model.yaml",
):
    os.makedirs(output_dir, exist_ok=True)
    cfg = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(model_defaults),
            OmegaConf.load(dataset_config),
            OmegaConf.load(cfg_path),
        ),
        resolve=True,
    )
    cfg["embeddings"] = normalize_embedding_configs(cfg.get("embeddings", []))
    
    prob_cfg = cfg.get("probing", {})
    # Tasks: speaker (class), noise (reg), mos (reg), semantic (reg/class)
    tasks = prob_cfg.get("tasks", ["mos"]) 
    emb_dim = cfg["embeddings"][0]["dim"]
    
    results = []
    
    for task_info in tasks:
        task_name = task_info if isinstance(task_info, str) else task_info["name"]
        target_col = task_info if isinstance(task_info, str) else task_info.get("target_column", task_name)
        task_type = "classification" if any(x in task_name for x in ["speaker", "lexical", "phoneme"]) else "regression"
        
        if "semantic" in task_name:
            task_type = task_info.get("type", "semantic_regression")

        print(f"\n>>> Rodando Probe: {task_name} ({task_type})")
        
        current_cfg = dict(cfg)
        current_cfg["probing"] = dict(current_cfg.get("probing", {}))
        current_cfg["probing"]["extra_columns"] = [target_col] if target_col != "mos" else []
        
        train_loader, val_loader, test_loader = build_loaders(current_cfg)
        
        # Pre-process labels if classification
        if task_type == "classification":
            le = LabelEncoder()
            all_labels = train_loader.dataset.df[target_col].astype(str).tolist() + \
                         val_loader.dataset.df[target_col].astype(str).tolist() + \
                         test_loader.dataset.df[target_col].astype(str).tolist()
            le.fit(all_labels)
            train_loader.dataset.df[target_col] = le.transform(train_loader.dataset.df[target_col].astype(str))
            val_loader.dataset.df[target_col]   = le.transform(val_loader.dataset.df[target_col].astype(str))
            test_loader.dataset.df[target_col]  = le.transform(test_loader.dataset.df[target_col].astype(str))
            out_dim = len(le.classes_)
        elif task_type == "semantic_regression":
            # Se for regressão semântica, o target é um vetor (ex: 384 dim)
            sample_target = train_loader.dataset[0][2][target_col]
            out_dim = len(sample_target) if hasattr(sample_target, "__len__") else 1
        else:
            out_dim = 1
            
        model = LinearProbe(emb_dim, out_dim)
        best_val = train_probe(model, train_loader, val_loader, task_type=task_type, target_col=target_col)
        
        # Test Evaluation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                embs, mos, masks, extras = batch
                x = embs[0].cuda()
                if x.dim() == 3: x = x.mean(dim=1)
                
                if task_type == "classification":
                    y = torch.tensor([e[target_col] for e in extras]).long()
                    out = model(x).argmax(dim=-1).cpu()
                elif task_type == "semantic_regression":
                    y = torch.tensor([e[target_col] for e in extras]).float()
                    out = model(x).cpu()
                else:
                    y = torch.tensor([e[target_col] for e in extras]).float() if target_col != "mos" else mos
                    out = model(x).squeeze(-1).cpu()
                
                all_preds.append(out.numpy())
                all_targets.append(y.numpy())
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        metrics = {"task": task_name, "embedding": cfg["embeddings"][0]["name"]}
        if task_type == "classification":
            metrics["accuracy"] = accuracy_score(all_targets, all_preds)
            metrics["f1_weighted"] = f1_score(all_targets, all_preds, average="weighted")
        elif task_type == "semantic_regression":
            cos_sims = [cosine_similarity(p, t) for p, t in zip(all_preds, all_targets)]
            metrics["cosine_sim"] = np.mean(cos_sims)
            metrics["pearson"] = np.mean([pearsonr(p, t)[0] for p, t in zip(all_preds.T, all_targets.T)])
        else:
            metrics["pearson"] = pearsonr(all_preds, all_targets)[0]
            metrics["r2"] = r2_score(all_targets, all_preds)
            metrics["mse"] = mean_squared_error(all_targets, all_preds)
            
        results.append(metrics)
        
    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, f"probing_results_{cfg['embeddings'][0]['name']}.csv")
    results_df.to_csv(output_path, index=False)
    print("\n--- Resultados Finais do Probing ---")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--output", default="results/probing")
    parser.add_argument("-d", "--dataset-config", default="configs/datasets/bvcc.yaml")
    parser.add_argument("-m", "--model-defaults", default="configs/model.yaml")
    args = parser.parse_args()
    run_probing_experiment(args.config, args.output, args.dataset_config, args.model_defaults)
