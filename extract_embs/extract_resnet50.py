"""
Extrai embeddings ResNet-50 (2048-dim) a partir de Mel-espectrogramas.
Mel-spec [128, T] -> resize (224, 224) -> 3-channel repeat -> ResNet-50 -> avgpool -> 2048d.
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
from torchvision import models, transforms

from extract_mel import compute_log_mel, SAMPLE_RATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalizacao ImageNet (aproximada para mel-spec)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_resnet50() -> nn.Module:
    """ResNet-50 pretrained ImageNet, sem a cabeca de classificacao."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove fc layer
    model.fc = nn.Identity()
    model = model.to(device).eval()
    return model


def extract_resnet50_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
) -> None:
    model = load_resnet50()
    os.makedirs(output_dir, exist_ok=True)
    skipped = 0
    errors  = 0

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    for filepath in tqdm(filelist, desc="ResNet-50"):
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

        # log-Mel spectrogram [128, T]
        log_mel = compute_log_mel(waveform, sr)       # 128, T

        # Normalizar para [0, 1] antes do resize
        mel_min = log_mel.min()
        mel_max = log_mel.max()
        if mel_max > mel_min:
            log_mel = (log_mel - mel_min) / (mel_max - mel_min)

        # Resize para (224, 224) via interpolacao bilinear
        log_mel = log_mel.unsqueeze(0).unsqueeze(0)    # [1, 1, 128, T]
        log_mel = nn.functional.interpolate(
            log_mel, size=(224, 224), mode="bilinear", align_corners=False
        )                                               # [1, 1, 224, 224]

        # Repetir para 3 canais (RGB-like)
        log_mel_3c = log_mel.repeat(1, 3, 1, 1)         # [1, 3, 224, 224]

        # Normalizar com mean/std do ImageNet
        log_mel_3c = normalize(log_mel_3c.squeeze(0))   # [3, 224, 224]
        log_mel_3c = log_mel_3c.unsqueeze(0).to(device) # [1, 3, 224, 224]

        with torch.no_grad():
            embedding = model(log_mel_3c)                # [1, 2048]

        torch.save(embedding.squeeze(0).cpu(), output_path)

    total = len(filelist)
    processed = total - skipped - errors
    print(f"\nResNet-50 concluido — processados: {processed} | "
          f"pulados: {skipped} | erros: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Extrai embeddings ResNet-50 de mel-espectrogramas")
    parser.add_argument("-b", "--base-dir",       required=True)
    parser.add_argument("-i", "--input-dir-name", required=True)
    parser.add_argument("-o", "--output-dir-name", default="resnet50_embeddings")
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

    extract_resnet50_embeddings(filelist, input_dir, output_dir)


if __name__ == "__main__":
    main()
