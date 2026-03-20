import os
import time
import argparse
import pandas as pd
import glob
from omegaconf import OmegaConf
from src.train import run_experiment

# Caminhos padrão dos CSVs de resultados
TRAIN_RESULTS_CSV = "train_results.csv"
TEST_RESULTS_CSV  = "test_results.csv"

def save_to_csv(row_dict, csv_path):
    """Adiciona uma linha ao CSV, criando-o se não existir."""
    df = pd.DataFrame([row_dict])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def main():
    parser = argparse.ArgumentParser(description="Executa todos os experimentos configurados.")
    parser.add_argument("-d", "--dataset-config", default="configs/datasets/bvcc.yaml", help="Configuração do dataset")
    parser.add_argument("-m", "--model-defaults", default="configs/model.yaml", help="Configuração base/default")
    parser.add_argument("-p", "--pattern", default="configs/experiments/ab*/*.yaml", help="Padrão de busca para experimentos")
    args = parser.parse_args()

    # 1. Carregar configurações base
    base_model_cfg = OmegaConf.load(args.model_defaults)
    dataset_cfg    = OmegaConf.load(args.dataset_config)

    # 2. Encontrar todos os arquivos de experimento
    experiment_files = glob.glob(args.pattern)
    print(f"Encontrados {len(experiment_files)} experimentos para rodar.")

    for exp_file in sorted(experiment_files):
        print(f"\n{'='*60}")
        print(f"Iniciando: {exp_file}")
        
        # Carregar config do experimento e mesclar com as bases
        exp_cfg = OmegaConf.load(exp_file)
        
        # Composição manual das configs (simulando Hydra)
        cfg = OmegaConf.merge(base_model_cfg, dataset_cfg, exp_cfg)
        
        # Converter para dict puro para compatibilidade com src.train
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        start_time = time.time()
        status = "success"
        
        try:
            # 3. Rodar experimento
            # run_experiment retorna (train_rows, eval_row)
            _, eval_row = run_experiment(cfg_dict)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 4. Preparar dados para train_results.csv
            train_record = {
                "experiment_name":  eval_row["experiment"],
                "dataset":          dataset_cfg.datasets.name,
                "config_path":      exp_file,
                "checkpoint_path":  eval_row["checkpoint_path"],
                "training_time_sec": round(duration, 2),
                "status":           status,
                "model_type":       eval_row["model_type"],
                "embeddings":       eval_row["embeddings"]
            }
            save_to_csv(train_record, TRAIN_RESULTS_CSV)
            
            # 5. Preparar dados para test_results.csv
            test_record = {
                "experiment_name":  eval_row["experiment"],
                "test_mse":         eval_row["test_mse"],
                "test_pearson":     eval_row["test_pearson"],
                "test_spearman":    eval_row["test_spearman"],
                "val_spearman":     eval_row["best_val_spearman"],
                "epochs_trained":   eval_row["epochs_trained"],
                "model_type":       eval_row["model_type"]
            }
            save_to_csv(test_record, TEST_RESULTS_CSV)
            
            print(f"Finalizado com sucesso em {duration:.2f}s")
            
        except Exception as e:
            print(f"ERRO no experimento {exp_file}: {e}")
            duration = time.time() - start_time
            error_record = {
                "experiment_name":  os.path.basename(exp_file),
                "dataset":          dataset_cfg.datasets.name,
                "config_path":      exp_file,
                "checkpoint_path":  "FAILED",
                "training_time_sec": round(duration, 2),
                "status":           f"error: {str(e)[:100]}",
                "model_type":       "?",
                "embeddings":       "?"
            }
            save_to_csv(error_record, TRAIN_RESULTS_CSV)

if __name__ == "__main__":
    main()