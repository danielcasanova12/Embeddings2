import os
import glob
import json
import traceback
import argparse
from datetime import datetime
from typing import List, Dict
from os.path import join, exists, relpath, splitext

import pandas as pd
import whisper
import torch

from extract_embs.extract_whisper_embeddings import extract_whisper_embeddings
from extract_embs.extract_contentvec import extract_contentvec_embeddings
from extract_embs.extract_speaker_embeddings import extract_speaker_embeddings
from extract_embs.extract_f0 import extract_f0_embeddings
from extract_embs.extract_hubert_embeddings import extract_hubert_embeddings
from extract_embs.extract_wavlm_embeddings import extract_wavlm_embeddings
from extract_embs.extract_wav2vec2_embeddings import extract_wav2vec2_embeddings


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


def run_extraction_on_csv(csv_path, base_dir, input_dir_name, output_base, column_name, asr_model, report_dir, suffix):
    # Relatório
    report = ExtractionReport(csv_path=csv_path, output_base=output_base)
    os.makedirs(report_dir, exist_ok=True)

    input_dir   = join(base_dir, input_dir_name)
    
    # 1. Coletar lista de arquivos
    filelist: List[str] = []
    df = pd.read_csv(csv_path)
    for f in df[column_name]:
        full = f if os.path.isabs(str(f)) else join(input_dir, str(f))
        filelist.append(full)

    # Pastas de saída
    output_whisper = join(output_base, "whisper")
    output_content = join(output_base, "contentvec")
    output_speaker = join(output_base, "speaker")
    output_f0      = join(output_base, "f0")
    output_hubert  = join(output_base, "hubert")
    output_wavlm   = join(output_base, "wavlm")
    output_wav2vec2 = join(output_base, "wav2vec2")

    print(f"\nTotal de arquivos a processar ({csv_path}): {len(filelist)}")

    # [0/8] ASR Transcription
    print(f"\n--- [0/8] Transcrição ASR: Whisper ({asr_model}) ---")
    report.begin_step("transcription")
    transcripts = []
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(asr_model, device=device)
        from tqdm import tqdm
        for filepath in tqdm(filelist, desc="Transcrevendo"):
            res = model.transcribe(filepath, fp16=(device=="cuda"))
            transcripts.append(res["text"].strip().lower())
        df["transcript"] = transcripts
        report.ok_step("transcription")
    except Exception as e:
        report.fail_step("transcription", e)
        print(f"ERRO FATAL em Transcrição: {e}")
        if "transcript" not in df.columns: df["transcript"] = [""] * len(filelist)

    # [1/8] Whisper Embeddings
    print("\n--- [1/8] Extração: Whisper Embeddings ---")
    report.begin_step("whisper")
    try:
        extract_whisper_embeddings(filelist, input_dir, output_whisper, "whisper-medium.en")
        verify_outputs(filelist, input_dir, output_whisper, report, "whisper")
        report.ok_step("whisper")
    except Exception as e:
        report.fail_step("whisper", e)

    # [2/8] ContentVec
    print("\n--- [2/8] Extração: ContentVec ---")
    report.begin_step("contentvec")
    try:
        extract_contentvec_embeddings(filelist, input_dir, output_content, "contentvec-best", layer=-1, pool=False)
        verify_outputs(filelist, input_dir, output_content, report, "contentvec")
        report.ok_step("contentvec")
    except Exception as e:
        report.fail_step("contentvec", e)

    # [3/8] Speaker ECAPA-TDNN
    print("\n--- [3/8] Extração: Speaker (ECAPA-TDNN) ---")
    report.begin_step("speaker")
    try:
        extract_speaker_embeddings(filelist, input_dir, output_speaker, "ecapa-tdnn", aggregate="mean", normalize=True)
        verify_outputs(filelist, input_dir, output_speaker, report, "speaker")
        report.ok_step("speaker")
    except Exception as e:
        report.fail_step("speaker", e)

    # [4/8] F0 CREPE
    print("\n--- [4/8] Extração: F0 (CREPE) ---")
    report.begin_step("f0")
    try:
        extract_f0_embeddings(filelist, input_dir, output_f0, backend="crepe",
                              hop_length=160, quantize=False, n_bins=256, f_min=50, f_max=1100)
        verify_outputs(filelist, input_dir, output_f0, report, "f0")
        report.ok_step("f0")
    except Exception as e:
        report.fail_step("f0", e)

    # [5/8] HuBERT
    print("\n--- [5/8] Extração: HuBERT ---")
    report.begin_step("hubert")
    try:
        extract_hubert_embeddings(filelist, input_dir, output_hubert, "hubert-base", layer=-1, pool=False)
        verify_outputs(filelist, input_dir, output_hubert, report, "hubert")
        report.ok_step("hubert")
    except Exception as e:
        report.fail_step("hubert", e)

    # [6/8] WavLM
    print("\n--- [6/8] Extração: WavLM ---")
    report.begin_step("wavlm")
    try:
        extract_wavlm_embeddings(filelist, input_dir, output_wavlm, "wavlm-base-plus", layer=-1, pool=False)
        verify_outputs(filelist, input_dir, output_wavlm, report, "wavlm")
        report.ok_step("wavlm")
    except Exception as e:
        report.fail_step("wavlm", e)

    # [7/8] wav2vec 2.0
    print("\n--- [7/8] Extração: wav2vec 2.0 ---")
    report.begin_step("wav2vec2")
    try:
        extract_wav2vec2_embeddings(filelist, input_dir, output_wav2vec2, "wav2vec2-base", layer=-1, pool=False)
        verify_outputs(filelist, input_dir, output_wav2vec2, report, "wav2vec2")
        report.ok_step("wav2vec2")
    except Exception as e:
        report.fail_step("wav2vec2", e)

    # Atualizar CSV
    def get_emb_path(audio_path, out_dir, input_dir):
        rel_p    = os.path.relpath(audio_path, input_dir)
        emb_file = splitext(rel_p)[0] + ".pt"
        return join(out_dir, emb_file)

    df["whisper_path"]    = [get_emb_path(f, output_whisper, input_dir) for f in filelist]
    df["contentvec_path"] = [get_emb_path(f, output_content, input_dir) for f in filelist]
    df["speaker_path"]    = [get_emb_path(f, output_speaker, input_dir) for f in filelist]
    df["f0_path"]         = [get_emb_path(f, output_f0,      input_dir) for f in filelist]
    df["hubert_path"]     = [get_emb_path(f, output_hubert,  input_dir) for f in filelist]
    df["wavlm_path"]      = [get_emb_path(f, output_wavlm,   input_dir) for f in filelist]
    df["wav2vec2_path"]   = [get_emb_path(f, output_wav2vec2, input_dir) for f in filelist]

    new_csv = csv_path.replace(".csv", suffix)
    df.to_csv(new_csv, index=False)
    print(f"CSV salvo em: {new_csv}")

    report.finalize()
    report.print_summary()
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(join(report_dir, f"report_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extrai todos os embeddings e Transcrições ASR."
    )
    parser.add_argument("-b", "--base-dir",       help="Diretório base do dataset")
    parser.add_argument("-i", "--input-dir-name",  help="Subpasta de áudios dentro do base-dir")
    parser.add_argument("-o", "--output-base",     default="embeddings")
    parser.add_argument("-c", "--csv-path",       help="CSV com lista de arquivos")
    parser.add_argument("-col", "--column-name",   default="filename")
    parser.add_argument("--suffix",                default="_with_embs.csv")
    parser.add_argument("--report-dir",            default="reports")
    parser.add_argument("--asr-model",             default="medium")
    
    # Novas opções para automação
    parser.add_argument("--dataset-config", help="Caminho para um YAML de dataset")
    parser.add_argument("--all-datasets", action="store_true", help="Processa todos os YAMLs em configs/datasets/")
    
    args = parser.parse_args()

    from omegaconf import OmegaConf
    
    configs_to_process = []
    if args.all_datasets:
        configs_to_process = glob.glob("configs/datasets/*.yaml")
    elif args.dataset_config:
        configs_to_process = [args.dataset_config]

    if configs_to_process:
        for cfg_path in configs_to_process:
            print(f"\nProcessing dataset config: {cfg_path}")
            cfg = OmegaConf.load(cfg_path)
            ds_name = cfg.datasets.name
            
            for split in ["train", "val", "test"]:
                if split in cfg.datasets:
                    csv_path = cfg.datasets[split].metadata_path
                    if not exists(csv_path):
                        print(f"Aviso: CSV não encontrado para {ds_name} {split}: {csv_path}")
                        continue
                    
                    # Tenta inferir base_dir e input_dir
                    # Frequentemente o base_dir é o pai da pasta 'sets' ou onde o CSV está
                    base_dir = os.path.dirname(os.path.dirname(csv_path)) 
                    # Se o CSV estiver na raiz do dataset, o acima pode estar errado.
                    # Vamos usar o diretório do CSV como fallback.
                    if not exists(base_dir): base_dir = os.path.dirname(csv_path)
                    
                    output_base = f"embeddings/{ds_name}/{split}"
                    
                    run_extraction_on_csv(
                        csv_path       = csv_path,
                        base_dir       = base_dir,
                        input_dir_name = ".",
                        output_base    = output_base,
                        column_name    = cfg.datasets[split].get("column_name", args.column_name),
                        asr_model      = args.asr_model,
                        report_dir     = join(args.report_dir, f"{ds_name}_{split}"),
                        suffix         = args.suffix
                    )
    else:
        # Modo manual (Compatibilidade com extract.sh)
        if not args.base_dir or not args.input_dir_name or not args.csv_path:
            print("Erro: Forneça --base-dir, --input-dir-name e --csv-path ou use --dataset-config / --all-datasets")
            return
            
        run_extraction_on_csv(
            csv_path       = args.csv_path,
            base_dir       = args.base_dir,
            input_dir_name = args.input_dir_name,
            output_base    = args.output_base,
            column_name    = args.column_name,
            asr_model      = args.asr_model,
            report_dir     = join(args.report_dir, os.path.basename(args.csv_path).replace(".csv", "")),
            suffix         = args.suffix
        )


if __name__ == "__main__":
    main()