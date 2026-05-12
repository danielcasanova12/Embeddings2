import os
import glob
import json
import traceback
import argparse
from datetime import datetime
from typing import List, Dict
from os.path import join, exists, relpath, splitext

import pandas as pd

from extract_embs.extract_whisper_embeddings import extract_whisper_embeddings
from extract_embs.extract_contentvec import extract_contentvec_embeddings
from extract_embs.extract_speaker_embeddings import extract_speaker_embeddings
from extract_embs.extract_f0 import extract_f0_embeddings


# ---------------------------------------------------------------------------
# Relatório de execução
# ---------------------------------------------------------------------------

class ExtractionReport:
    def __init__(self, csv_path: str, output_base: str):
        self.csv_path    = csv_path
        self.output_base = output_base
        self.started_at  = datetime.now().isoformat(timespec="seconds")
        self.finished_at = None
        self.steps: Dict[str, dict] = {}

    def begin_step(self, name: str):
        self.steps[name] = {"status": "running", "error": None, "missing_files": []}

    def ok_step(self, name: str):
        self.steps[name]["status"] = "ok"

    def fail_step(self, name: str, exc: Exception):
        self.steps[name]["status"] = "error"
        self.steps[name]["error"]  = traceback.format_exc()

    def add_missing(self, step: str, path: str):
        self.steps[step]["missing_files"].append(path)

    def finalize(self):
        self.finished_at = datetime.now().isoformat(timespec="seconds")

    def to_dict(self) -> dict:
        summary = {}
        for name, info in self.steps.items():
            summary[name] = {
                "status":        info["status"],
                "error":         info["error"],
                "missing_count": len(info["missing_files"]),
                "missing_files": info["missing_files"],
            }
        return {
            "csv_path":    self.csv_path,
            "output_base": self.output_base,
            "started_at":  self.started_at,
            "finished_at": self.finished_at,
            "steps":       summary,
        }

    def print_summary(self):
        print("\n" + "=" * 70)
        print("RELATÓRIO DE EXTRAÇÃO")
        print(f"  CSV       : {self.csv_path}")
        print(f"  Início    : {self.started_at}")
        print(f"  Fim       : {self.finished_at}")
        print("-" * 70)
        for name, info in self.steps.items():
            status_icon = "✓" if info["status"] == "ok" else "✗"
            n_missing   = len(info["missing_files"])
            print(f"  [{status_icon}] {name:<25} | status={info['status']:<7} | faltando={n_missing}")
            if info["error"]:
                # Mostra apenas a última linha do traceback
                last_line = info["error"].strip().splitlines()[-1]
                print(f"       ERRO: {last_line}")
            if n_missing:
                for p in info["missing_files"][:5]:
                    print(f"       FALTANDO: {p}")
                if n_missing > 5:
                    print(f"       ... e mais {n_missing - 5} arquivos")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Verificação pós-extração
# ---------------------------------------------------------------------------

def verify_outputs(filelist: List[str], input_dir: str, output_dir: str, report: ExtractionReport, step_name: str):
    """Verifica se todos os .pt foram gerados e registra os faltantes."""
    for filepath in filelist:
        rel_p = relpath(filepath, input_dir)
        expected = join(output_dir, splitext(rel_p)[0] + ".pt")
        if not exists(expected):
            report.add_missing(step_name, expected)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrai todos os embeddings (Whisper, ContentVec, Speaker, F0) com relatório de erros."
    )
    parser.add_argument("-b", "--base-dir",       required=True)
    parser.add_argument("-i", "--input-dir-name",  required=True)
    parser.add_argument("-o", "--output-base",     default="embeddings")
    parser.add_argument("-c", "--csv-path")
    parser.add_argument("-col", "--column-name",   default="filename")
    parser.add_argument("--suffix",                default="_with_embs.csv")
    parser.add_argument("--report-dir",            default=None,
                        help="Pasta para salvar o JSON de relatório (padrão: output_base)")
    args = parser.parse_args()

    input_dir   = join(args.base_dir, args.input_dir_name)
    output_base = args.output_base if os.path.isabs(args.output_base) else join(args.base_dir, args.output_base)
    report_dir  = args.report_dir or output_base
    os.makedirs(report_dir, exist_ok=True)

    # Relatório
    report = ExtractionReport(csv_path=args.csv_path or "glob", output_base=output_base)

    # ------------------------------------------------------------------
    # 1. Coletar lista de arquivos
    # ------------------------------------------------------------------
    filelist: List[str] = []
    df = None

    if args.csv_path and exists(args.csv_path):
        df = pd.read_csv(args.csv_path)
        for f in df[args.column_name]:
            full = f if os.path.isabs(str(f)) else join(input_dir, str(f))
            filelist.append(full)
    else:
        print(f"Buscando arquivos .wav em: {input_dir}")
        filelist = glob.glob(join(input_dir, "**", "*.wav"), recursive=True)
        if not filelist:
            print("Nenhum arquivo .wav encontrado!")
            return
        df = pd.DataFrame({args.column_name: filelist})

    # Pastas de saída
    output_whisper = join(output_base, "whisper")
    output_content = join(output_base, "contentvec")
    output_speaker = join(output_base, "speaker")
    output_f0      = join(output_base, "f0")

    print(f"\nTotal de arquivos a processar: {len(filelist)}")

    # ------------------------------------------------------------------
    # [1/4] Whisper
    # ------------------------------------------------------------------
    print("\n--- [1/4] Extração: Whisper ---")
    report.begin_step("whisper")
    try:
        extract_whisper_embeddings(filelist, input_dir, output_whisper, "whisper-large-v3")
        verify_outputs(filelist, input_dir, output_whisper, report, "whisper")
        report.ok_step("whisper")
    except Exception as e:
        report.fail_step("whisper", e)
        print(f"ERRO FATAL em Whisper: {e}")

    # ------------------------------------------------------------------
    # [2/4] ContentVec
    # ------------------------------------------------------------------
    print("\n--- [2/4] Extração: ContentVec ---")
    report.begin_step("contentvec")
    try:
        extract_contentvec_embeddings(filelist, input_dir, output_content, "contentvec-best", layer=-1, pool=False)
        verify_outputs(filelist, input_dir, output_content, report, "contentvec")
        report.ok_step("contentvec")
    except Exception as e:
        report.fail_step("contentvec", e)
        print(f"ERRO FATAL em ContentVec: {e}")

    # ------------------------------------------------------------------
    # [3/4] Speaker ECAPA-TDNN
    # ------------------------------------------------------------------
    print("\n--- [3/4] Extração: Speaker (ECAPA-TDNN) ---")
    report.begin_step("speaker")
    try:
        extract_speaker_embeddings(filelist, input_dir, output_speaker, "ecapa-tdnn", aggregate="mean", normalize=True)
        verify_outputs(filelist, input_dir, output_speaker, report, "speaker")
        report.ok_step("speaker")
    except Exception as e:
        report.fail_step("speaker", e)
        print(f"ERRO FATAL em Speaker: {e}")

    # ------------------------------------------------------------------
    # [4/4] F0 CREPE
    # ------------------------------------------------------------------
    print("\n--- [4/4] Extração: F0 (CREPE) ---")
    report.begin_step("f0")
    try:
        extract_f0_embeddings(filelist, input_dir, output_f0, backend="crepe",
                              hop_length=160, quantize=False, n_bins=256, f_min=50, f_max=1100)
        verify_outputs(filelist, input_dir, output_f0, report, "f0")
        report.ok_step("f0")
    except Exception as e:
        report.fail_step("f0", e)
        print(f"ERRO FATAL em F0: {e}")

    # ------------------------------------------------------------------
    # Atualizar CSV
    # ------------------------------------------------------------------
    print("\n--- Atualizando CSV com caminhos dos embeddings ---")

    def get_emb_path(audio_path, out_dir, input_dir):
        rel_p    = os.path.relpath(audio_path, input_dir)
        emb_file = splitext(rel_p)[0] + ".pt"
        return join(out_dir, emb_file)

    df["whisper_path"]    = [get_emb_path(f, output_whisper, input_dir) for f in filelist]
    df["contentvec_path"] = [get_emb_path(f, output_content, input_dir) for f in filelist]
    df["speaker_path"]    = [get_emb_path(f, output_speaker, input_dir) for f in filelist]
    df["f0_path"]         = [get_emb_path(f, output_f0,      input_dir) for f in filelist]

    if args.csv_path:
        new_csv = args.csv_path.replace(".csv", args.suffix)
    else:
        new_csv = join(args.base_dir, "dataset" + args.suffix)
    df.to_csv(new_csv, index=False)
    print(f"CSV salvo em: {new_csv}")

    # ------------------------------------------------------------------
    # Relatório final
    # ------------------------------------------------------------------
    report.finalize()
    report.print_summary()

    # Salva JSON
    tag         = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = join(report_dir, f"report_{tag}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
    print(f"\nRelatório JSON salvo em: {report_path}")


if __name__ == "__main__":
    main()