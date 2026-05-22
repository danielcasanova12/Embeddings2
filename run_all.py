import os
import time
import argparse
import pandas as pd
import glob
import datetime
import subprocess
from omegaconf import OmegaConf

# ReportLab imports
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image

from src.train import run_experiment

# Caminhos padrão dos CSVs de resultados
TRAIN_RESULTS_CSV = "train_results.csv"
TEST_RESULTS_CSV  = "test_results.csv"

def save_to_csv(row_dict, csv_path):
    df = pd.DataFrame([row_dict])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def run_command(cmd, desc):
    print(f"\n>>> Executando {desc}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def gerar_relatorio_pdf(resultados_gerais, resultados_analise, caminho_saida):
    """Gera um relatório PDF completo com tabelas e referências a plots."""
    doc = SimpleDocTemplate(str(caminho_saida), pagesize=landscape(letter))
    elementos = []
    estilos = getSampleStyleSheet()

    elementos.append(Paragraph("Relatório Consolidado de Experimentos MOS", estilos['Title']))
    elementos.append(Spacer(1, 12))

    # --- Seção 1: Experimentos de Treino (1, 2, 3, 8) ---
    elementos.append(Paragraph("1. Experimentos de Treino (Baseline, Fusion, Interaction, Pooling)", estilos['Heading2']))
    if resultados_gerais:
        cabecalhos = list(resultados_gerais[0].keys())
        dados_tabela = [[Paragraph(f"<b>{c}</b>", estilos['Normal']) for c in cabecalhos]]
        for linha in resultados_gerais:
            dados_tabela.append([Paragraph(str(linha.get(c, "")), estilos['Normal']) for c in cabecalhos])

        t = Table(dados_tabela)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elementos.append(t)

    elementos.append(Spacer(1, 24))

    # --- Seção 2: Experimentos de Análise (4, 5, 6, 7) ---
    elementos.append(Paragraph("2. Experimentos de Análise e Robustez", estilos['Heading2']))
    for analise in resultados_analise:
        elementos.append(Paragraph(f"<b>{analise['titulo']}</b>", estilos['Heading3']))
        elementos.append(Paragraph(analise['descricao'], estilos['Normal']))

        if 'csv' in analise and os.path.exists(analise['csv']):
            df_an = pd.read_csv(analise['csv'])
            data = [df_an.columns.tolist()] + df_an.values.tolist()
            t_an = Table(data)
            t_an.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.grey)]))
            elementos.append(t_an)

        if 'plots' in analise:
            for p in analise['plots']:
                if os.path.exists(p):
                    elementos.append(Spacer(1, 12))
                    # Ajusta tamanho da imagem para caber no PDF
                    img = Image(p, width=400, height=300)
                    elementos.append(img)
                    elementos.append(Paragraph(f"Figura: {os.path.basename(p)}", estilos['Italic']))

        elementos.append(Spacer(1, 12))

    doc.build(elementos)

def main():
    parser = argparse.ArgumentParser(description="Executa o pipeline completo de 8 experimentos.")
    parser.add_argument("-d", "--dataset-config", default="configs/datasets/bvcc.yaml", help="Configuração do dataset")
    parser.add_argument("-m", "--model-defaults", default="configs/model.yaml", help="Configuração base/default")
    args = parser.parse_args()

    base_model_cfg = OmegaConf.load(args.model_defaults)
    dataset_cfg    = OmegaConf.load(args.dataset_config)
    dataset_name   = dataset_cfg.datasets.name

    pdf_results_train = []
    analysis_results = []

    # --- FASE 1: Experimentos 1, 2, 3, 8 (Treinamento) ---
    # Busca por configs que sigam o padrão exp1, exp2, exp3, exp8
    pattern = "configs/experiments/{exp1,exp2,exp3,exp8}/**/*.yaml"
    import glob
    # Nota: Windows glob pode não suportar {} nativamente, vamos fazer manual
    exp_dirs = ["exp1", "exp2", "exp3", "exp8"]
    experiment_files = []
    for ed in exp_dirs:
        experiment_files.extend(glob.glob(f"configs/experiments/{ed}/**/*.yaml", recursive=True))

    print(f"Rodando {len(experiment_files)} experimentos de treinamento...")

    for exp_file in sorted(experiment_files):
        print(f"\nIniciando Treino: {exp_file}")
        exp_cfg = OmegaConf.load(exp_file)
        cfg = OmegaConf.merge(base_model_cfg, dataset_cfg, exp_cfg)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        try:
            _, eval_row = run_experiment(cfg_dict)
            pdf_results_train.append({
                "ID": eval_row["experiment"],
                "Tipo": eval_row["model_type"],
                "Embeddings": eval_row["embeddings"],
                "Pearson": f"{eval_row.get('test_pearson', 0):.4f}",
                "Spearman": f"{eval_row.get('test_spearman', 0):.4f}"
            })
        except Exception as e:
            print(f"Erro no treino {exp_file}: {e}")

    # --- FASE 2: Experimento 4 (Zero-Shot) ---
    print("\n--- [Exp 4] Zero-Shot Evaluation ---")
    # Assume que temos os checkpoints dos experimentos FULL e minus_c do exp2
    # Procurar o melhor checkpoint (simplificado)
    ckpt_full = glob.glob("checkpoints/exp2_full/**/*.ckpt", recursive=True)
    ckpt_minc = glob.glob("checkpoints/exp2_minus_c/**/*.ckpt", recursive=True)

    if ckpt_full and ckpt_minc:
        run_command([
            "python", "scripts/exp4_zeroshot.py",
            "--full", ckpt_full[0],
            "--minc", ckpt_minc[0],
            "-c", args.dataset_config # Avalia no próprio dataset como teste ou outros
        ], "Exp 4")
        analysis_results.append({
            "titulo": "Experimento 4: Zero-Shot Generalization",
            "descricao": "Comparação de robustez entre modelos FULL e -C (sem lexical) em novos domínios.",
            "csv": "results/zeroshot/zeroshot_results.csv"
        })

    # --- FASE 3: Experimento 5 (Probing) ---
    print("\n--- [Exp 5] Linear Probing ---")
    # Rodar probing para um dos modelos, ex: whisper
    run_command([
        "python", "scripts/exp5_probing.py",
        "-c", "configs/experiments/exp1/whisper.yaml"
    ], "Exp 5")
    analysis_results.append({
        "titulo": "Experimento 5: Probing Linear Analysis",
        "descricao": "Análise de quanta informação de locutor, ruído e conteúdo está codificada linearmente.",
        "csv": "results/probing/probing_results_whisper.csv"
    })

    # --- FASE 4: Experimento 6 (UMAP) ---
    print("\n--- [Exp 6] UMAP Visualization ---")
    if ckpt_minc:
        run_command([
            "python", "scripts/exp6_umap.py",
            "-k", ckpt_minc[0],
            "-c", args.dataset_config
        ], "Exp 6")
        analysis_results.append({
            "titulo": "Experimento 6: UMAP Latent Space",
            "descricao": "Visualização do espaço latente colorido por MOS e por Dataset.",
            "plots": ["results/umap/umap_mos.png", "results/umap/umap_datasets.png"]
        })

    # --- FASE 5: Experimento 7 (Perturbation) ---
    print("\n--- [Exp 7] Semantic Perturbation ---")
    # Primeiro gerar pares (requer que o extract_all tenha gerado o _with_embs.csv com transcrição)
    dataset_with_trans = args.dataset_config.replace(".yaml", "").split("/")[-1] + "_with_embs.csv"
    # Nota: O caminho real depende de onde extract_all salvou. Assume local ou base-dir.
    if os.path.exists(dataset_with_trans):
        run_command([
            "python", "scripts/prepare_perturbation_pairs.py",
            "-i", dataset_with_trans
        ], "Prep Exp 7")
        run_command([
            "python", "scripts/exp7_perturbation.py",
            "-i", "perturbation_pairs.csv"
        ], "Exp 7")
        analysis_results.append({
            "titulo": "Experimento 7: Semantic Perturbation Analysis",
            "descricao": "Medição da sensibilidade dos embeddings a variações de conteúdo vs qualidade.",
            "csv": "results/perturbation/perturbation_results.csv"
        })

    # --- GERAÇÃO DO PDF FINAL ---
    data_atual = datetime.datetime.now().strftime("%Y-%m-%d")
    pdf_filename = f"FULL_REPORT_{dataset_name}_{data_atual}.pdf"
    gerar_relatorio_pdf(pdf_results_train, analysis_results, pdf_filename)
    print(f"\nPipeline Completo Finalizado! Relatório: {pdf_filename}")

if __name__ == "__main__":
    main()