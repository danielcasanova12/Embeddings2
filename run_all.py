import os
import time
import argparse
import pandas as pd
import glob
import datetime
from omegaconf import OmegaConf

# ReportLab imports
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

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

def gerar_relatorio_pdf(resultados, caminho_saida):
    """Gera um relatório em PDF contendo a tabela de resultados dos experimentos."""
    if not resultados:
        return

    # Usar modo paisagem para melhor acomodação das colunas
    doc = SimpleDocTemplate(str(caminho_saida), pagesize=landscape(letter))
    elementos = []
    estilos = getSampleStyleSheet()
    estilo_normal = estilos['Normal']
    
    titulo = Paragraph("Resumo dos Experimentos", estilos['Title'])
    elementos.append(titulo)
    elementos.append(Spacer(1, 12))

    # Obter cabeçalhos das chaves do primeiro dicionário
    cabecalhos = list(resultados[0].keys())
    
    dados_tabela = []
    
    # Linha de cabeçalho
    linha_cabecalho = [Paragraph(f"<b>{c}</b>", estilo_normal) for c in cabecalhos]
    dados_tabela.append(linha_cabecalho)

    # Linhas de dados
    for linha in resultados:
        linha_dados = [Paragraph(str(linha.get(c, "")), estilo_normal) for c in cabecalhos]
        dados_tabela.append(linha_dados)

    tabela = Table(dados_tabela)

    estilo_tabela = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
    tabela.setStyle(estilo_tabela)
    elementos.append(tabela)

    doc.build(elementos)

def main():
    parser = argparse.ArgumentParser(description="Executa todos os experimentos configurados.")
    parser.add_argument("-d", "--dataset-config", default="configs/datasets/bvcc.yaml", help="Configuração do dataset")
    parser.add_argument("-m", "--model-defaults", default="configs/model.yaml", help="Configuração base/default")
    parser.add_argument("-p", "--pattern", default="configs/experiments/**/*.yaml", help="Padrão de busca para experimentos")
    args = parser.parse_args()

    # 1. Carregar configurações base
    base_model_cfg = OmegaConf.load(args.model_defaults)
    dataset_cfg    = OmegaConf.load(args.dataset_config)

    # 2. Encontrar todos os arquivos de experimento
    experiment_files = glob.glob(args.pattern, recursive=True)
    print(f"Encontrados {len(experiment_files)} experimentos para rodar.")

    # Lista para acumular os resultados para o PDF
    pdf_results = []

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
            
            # 6. Preparar dados resumidos para o PDF
            pdf_results.append({
                "Experimento": eval_row["experiment"],
                "Modelo": eval_row["model_type"],
                "Tempo (s)": f"{duration:.2f}",
                "Status": "Sucesso",
                "Test MSE": f"{eval_row.get('test_mse', 0):.4f}",
                "Test Pearson": f"{eval_row.get('test_pearson', 0):.4f}",
                "Test Spearman": f"{eval_row.get('test_spearman', 0):.4f}"
            })
            
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
            
            # Registrar erro para o PDF
            pdf_results.append({
                "Experimento": os.path.basename(exp_file),
                "Modelo": "-",
                "Tempo (s)": f"{duration:.2f}",
                "Status": "Erro",
                "Test MSE": "-",
                "Test Pearson": "-",
                "Test Spearman": "-"
            })

    # Após rodar todos os experimentos, gerar o PDF
    if pdf_results:
        # Obter nome do dataset e data atual
        dataset_name = dataset_cfg.datasets.name
        data_atual = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Montar o nome do arquivo dinâmico
        pdf_filename = f"{dataset_name}-{data_atual}.pdf"
        
        # Gerar e salvar o PDF
        gerar_relatorio_pdf(pdf_results, pdf_filename)
        print(f"\n{'='*60}")
        print(f"Relatório final em PDF salvo com sucesso em: {pdf_filename}")

if __name__ == "__main__":
    main()