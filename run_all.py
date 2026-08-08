import os
import time
import argparse
import pandas as pd
import glob
import datetime
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

# ReportLab imports
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image

from src.train import run_experiment

TRAIN_RESULTS_CSV = "train_results.csv"
TEST_RESULTS_CSV  = "test_results.csv"

# -----------------------------------------------------------------------
# Novo: resolve paths dos CSVs a partir do novo padrão de pastas
# -----------------------------------------------------------------------
EMBEDDINGS_ROOT = Path("embeddings")  # mantém para compatibilidade

def _resolve_case_insensitive_path(path_str: str) -> Path:
    """
    Resolve um path existente mesmo quando há diferença de caixa
    entre config e filesystem (ex.: bvcc vs BVCC em Linux).
    """
    original = Path(path_str)
    if original.exists():
        return original

    base = Path(original.anchor) if original.is_absolute() else Path(".")
    parts = [p for p in original.parts if p not in ("", original.anchor)]
    current = base

    for part in parts:
        if not current.exists():
            return original

        exact = current / part
        if exact.exists():
            current = exact
            continue

        lowered = part.lower()
        matches = [child for child in current.iterdir() if child.name.lower() == lowered]
        if len(matches) != 1:
            return original
        current = matches[0]

    return current


def _resolve_dataset_root(dataset_name: str) -> Path:
    return _resolve_case_insensitive_path(str(EMBEDDINGS_ROOT / dataset_name))

def resolve_data_paths(cfg_dict: dict) -> dict:
    """
    Usa os metadata_path já definidos no config (absolutos ou relativos).
    Se houver diferença de caixa entre config e filesystem, reescreve
    para o path real antes de validar.
    """
    datasets = cfg_dict.get("datasets", {})
    for split in ["train", "val", "test"]:
        split_cfg = datasets.get(split, {})
        path = split_cfg.get("metadata_path", "")
        if not path:
            continue

        resolved = _resolve_case_insensitive_path(path)
        split_cfg["metadata_path"] = str(resolved)

        if not resolved.exists():
            raise FileNotFoundError(
                f"metadata_path não encontrado para split '{split}': {path}"
            )
    return cfg_dict


def save_to_csv(row_dict, csv_path):
    df = pd.DataFrame([row_dict])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)


def run_command(cmd, desc):
    """Executa subcomando garantindo que src/ seja encontrado."""
    print(f"\n>>> Executando {desc}: {' '.join(cmd)}")
    env = os.environ.copy()
    project_root = str(Path(__file__).parent.resolve())
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)


def gerar_relatorio_pdf(resultados_gerais, resultados_analise, caminho_saida):
    """Gera um relatório PDF completo com tabelas e referências a plots."""
    doc = SimpleDocTemplate(str(caminho_saida), pagesize=landscape(letter))
    elementos = []
    estilos = getSampleStyleSheet()

    elementos.append(Paragraph("Relatório Consolidado de Experimentos MOS", estilos['Title']))
    elementos.append(Spacer(1, 12))

    elementos.append(Paragraph("1. Experimentos de Treino (Protocolo Completo 1-10)", estilos['Heading2']))
    if resultados_gerais:
        cabecalhos = ["ID", "Tipo", "Embeddings", "Pearson", "Spearman", "Detalhes Extra"]
        dados_tabela = [[Paragraph(f"<b>{c}</b>", estilos['Normal']) for c in cabecalhos]]
        for linha in resultados_gerais:
            dados_tabela.append([
                Paragraph(str(linha.get("ID", "")), estilos['Normal']),
                Paragraph(str(linha.get("Tipo", "")), estilos['Normal']),
                Paragraph(str(linha.get("Embeddings", "")), estilos['Normal']),
                Paragraph(str(linha.get("Pearson", "")), estilos['Normal']),
                Paragraph(str(linha.get("Spearman", "")), estilos['Normal']),
                Paragraph(str(linha.get("Detalhes Extra", "-")), estilos['Normal']),
            ])
        t = Table(dados_tabela, colWidths=[80, 100, 200, 60, 60, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elementos.append(t)

    elementos.append(Spacer(1, 24))

    pesos_exp = [r for r in resultados_gerais if r.get("Tipo") == "weighted_fusion"]
    if pesos_exp:
        elementos.append(Paragraph("1.1 Detalhamento de Pesos Dinâmicos (Exp 9)", estilos['Heading3']))
        for p in pesos_exp:
            if "weights" in p:
                elementos.append(Paragraph(f"Experimento: {p['ID']}", estilos['Normal']))
                w_data = [["Embedding", "Peso Médio"]] + [[k, f"{v:.4f}"] for k, v in p["weights"].items()]
                tw = Table(w_data)
                tw.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.grey)]))
                elementos.append(tw)
                elementos.append(Spacer(1, 12))

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
                    img = Image(p, width=400, height=300)
                    elementos.append(img)
                    elementos.append(Paragraph(f"Figura: {os.path.basename(p)}", estilos['Italic']))

        elementos.append(Spacer(1, 12))

    doc.build(elementos)


def main():
    parser = argparse.ArgumentParser(description="Executa o pipeline completo de experimentos.")
    parser.add_argument("-d", "--dataset-config", default="configs/datasets/bvcc.yaml")
    parser.add_argument("-m", "--model-defaults",  default="configs/model.yaml")
    args = parser.parse_args()

    base_model_cfg = OmegaConf.load(args.model_defaults)
    dataset_cfg    = OmegaConf.load(args.dataset_config)
    dataset_name   = dataset_cfg.datasets.name

    # Valida que os CSVs existem antes de começar
    dataset_root = _resolve_dataset_root(dataset_name)
    missing = []
    for split in ["train", "val", "test"]:
        p = _resolve_case_insensitive_path(str(dataset_root / split / "metadata_with_embs.csv"))
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("AVISO: CSVs de embeddings não encontrados:")
        for m in missing:
            print(f"  {m}")
        print("Execute o pipeline de extração antes de rodar os experimentos.\n")

    pdf_results_train = []
    analysis_results  = []

    # --- FASE 1: Treinamento (exp1, 2, 3, 8, 9, 10) ---
    exp_dirs = ["exp1", "exp2", "exp3", "exp8", "exp9", "exp10"]
    experiment_files = []
    for ed in exp_dirs:
        experiment_files.extend(glob.glob(f"configs/experiments/{ed}/**/*.yaml", recursive=True))

    print(f"Rodando {len(experiment_files)} experimentos de treinamento...")

    for exp_file in sorted(experiment_files):
        print(f"\nIniciando Treino: {exp_file}")
        exp_cfg = OmegaConf.load(exp_file)
        cfg = OmegaConf.merge(base_model_cfg, dataset_cfg, exp_cfg)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # ← ponto central do ajuste: reescreve os paths
        try:
            cfg_dict = resolve_data_paths(cfg_dict)
        except ValueError as e:
            print(f"  Skipping {exp_file}: {e}")
            continue

        try:
            _, eval_row = run_experiment(cfg_dict)
            pdf_results_train.append({
                "ID":        eval_row["experiment"],
                "Tipo":      eval_row["model_type"],
                "Embeddings": eval_row["embeddings"],
                "Pearson":   f"{eval_row.get('test_pearson', 0):.4f}",
                "Spearman":  f"{eval_row.get('test_spearman', 0):.4f}",
            })
        except Exception as e:
            print(f"Erro no treino {exp_file}: {e}")

    # --- FASE 2: Exp 4 (Zero-Shot) ---
    print("\n--- [Exp 4] Zero-Shot Evaluation ---")
    ckpt_full = glob.glob("checkpoints/exp2_full/**/*.ckpt", recursive=True)
    ckpt_minc = glob.glob("checkpoints/exp2_minus_c/**/*.ckpt", recursive=True)

    if ckpt_full and ckpt_minc:
        run_command([
            "python", "scripts/exp4_zeroshot.py",
            "--full", ckpt_full[0],
            "--minc", ckpt_minc[0],
            "-c", args.dataset_config,
        ], "Exp 4")
        analysis_results.append({
            "titulo":    "Experimento 4: Zero-Shot Generalization",
            "descricao": "Comparação de robustez entre modelos FULL e -C em novos domínios.",
            "csv":       "results/zeroshot/zeroshot_results.csv",
        })

    # --- FASE 3: Exp 5 (Probing) ---
    print("\n--- [Exp 5] Linear Probing ---")
    run_command([
        "python", "scripts/exp5_probing.py",
        "-c", "configs/experiments/exp1/whisper.yaml",
        "-d", args.dataset_config,
        "-m", args.model_defaults,
    ], "Exp 5")
    analysis_results.append({
        "titulo":    "Experimento 5: Probing Linear Analysis",
        "descricao": "Análise de quanta informação de locutor, ruído e conteúdo está codificada linearmente.",
        "csv":       "results/probing/probing_results_whisper.csv",
    })

    # --- FASE 4: Exp 6 (UMAP) ---
    print("\n--- [Exp 6] UMAP Visualization ---")
    if ckpt_minc:
        run_command([
            "python", "scripts/exp6_umap.py",
            "-k", ckpt_minc[0],
            "-c", args.dataset_config,
        ], "Exp 6")
        analysis_results.append({
            "titulo":    "Experimento 6: UMAP Latent Space",
            "descricao": "Visualização do espaço latente colorido por MOS e por Dataset.",
            "plots":     ["results/umap/umap_mos.png", "results/umap/umap_datasets.png"],
        })

    # --- FASE 5: Exp 7 (Perturbation) ---
    print("\n--- [Exp 7] Semantic Perturbation ---")
    dataset_with_trans = str(_resolve_case_insensitive_path(str(dataset_root / "test" / "metadata_with_embs.csv")))
    if os.path.exists(dataset_with_trans):
        run_command([
            "python", "scripts/prepare_perturbation_pairs.py",
            "-i", dataset_with_trans,
        ], "Prep Exp 7")
        run_command([
            "python", "scripts/exp7_perturbation.py",
            "-i", "perturbation_pairs.csv",
        ], "Exp 7")
        analysis_results.append({
            "titulo":    "Experimento 7: Semantic Perturbation Analysis",
            "descricao": "Sensibilidade dos embeddings a variações de conteúdo vs qualidade.",
            "csv":       "results/perturbation/perturbation_results.csv",
        })

    # --- PDF Final ---
    data_atual   = datetime.datetime.now().strftime("%Y-%m-%d")
    pdf_filename = f"FULL_REPORT_{dataset_name}_{data_atual}.pdf"
    gerar_relatorio_pdf(pdf_results_train, analysis_results, pdf_filename)
    print(f"\nPipeline Completo Finalizado! Relatório: {pdf_filename}")


if __name__ == "__main__":
    main()
