"""
Extrai embeddings AST (Audio Spectrogram Transformer) 768-dim.
Mel-spec [128, T] -> AST encoder -> mean pool -> 768d.
Usa MIT/ast-finetuned-audioset-10-10-0.4593 (SOTA AudioSet).
"""
import os
import glob
import argparse
from typing import List
from os.path import exists, join, relpath, dirname

import torch
import torchaudio
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import ASTModel, ASTFeatureExtractor

from extract_mel import compute_log_mel, SAMPLE_RATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


def load_ast() -> tuple:
    """Carrega AST pretrained (SOTA AudioSet)."""
    print(f"Carregando AST: {MODEL_ID} ...")
    model = ASTModel.from_pretrained(MODEL_ID)
    model = model.to(device).eval()
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    return model, feature_extractor


def extract_ast_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
) -> None:
    model, feature_extractor = load_ast()
    os.makedirs(output_dir, exist_ok=True)
    skipped = 0
    errors  = 0

    for filepath in tqdm(filelist, desc="AST"):
        if not exists(filepath):
            print(f"[AVISO] Arquivo nao encontrado: {filepath}")
            errors += 1
            continue

        rel_p       = relpath(filepath, input_dir)
        output_path = join(output_dir, os.path.splitext(rel_p)[0] + ".pt")
        os.makedirs(dirname(output_path), exist_ok=True)

        if exists(output_path):
            skipped += 1
            continue

        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception as e:
            print(f"[ERRO] Falha ao carregar {filepath}: {e}")
            errors += 1
            continue

        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample para 16 kHz
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform_np = waveform.squeeze().numpy()

        # AST feature extractor espera audio raw, extrai mel internamente
        inputs = feature_extractor(
            waveform_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        # AST padroniza para 1024 frames de tempo
        input_values = inputs["input_values"].to(device)   # [1, 1024, 128]

        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)

        # Mean pooling sobre a sequencia temporal (alternativa ao [CLS])
        last_hidden = outputs.last_hidden_state          # [1, seq_len, 768]
        embedding   = last_hidden.mean(dim=1).squeeze(0)  # [768]

        torch.save(embedding.cpu(), output_path)

    total = len(filelist)
    processed = total - skipped - errors
    print(f"\nAST concluido — processados: {processed} | "
          f"pulados: {skipped} | erros: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Extrai embeddings AST de mel-espectrogramas")
    parser.add_argument("-b", "--base-dir",       required=True)
    parser.add_argument("-i", "--input-dir-name", required=True)
    parser.add_argument("-o", "--output-dir-name", default="ast_embeddings")
    parser.add_argument("-c", "--input-csv")
    parser.add_argument("-col", "--column-name", default="filename")

    args = parser.parse_args()

    input_dir  = os.path.join(args.base_dir, args.input_dir_name)
    output_dir = os.path.join(args.base_dir, args.output_dir_name)

    if args.input_csv and exists(args.input_csv):
        df       = pd.read_csv(args.input_csv)
        filelist = df[args.column_name].tolist()
    else:
        filelist = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)

    extract_ast_embeddings(filelist, input_dir, output_dir)


if __name__ == "__main__":
    main()
