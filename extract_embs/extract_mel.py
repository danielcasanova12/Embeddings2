"""
Extrai log-Mel spectrogram (128 bandas) e salva como .pt.
Fluxo: waveform 16kHz -> STFT -> mel filterbank -> 128 mel bands -> log-mel -> .pt
"""
import os
import glob
import argparse
from typing import List, Optional
from os.path import exists, join, relpath, dirname

import torch
import torchaudio
import pandas as pd
from tqdm import tqdm

# Parâmetros fixos do mel-spectrograma (compatíveis ResNet-50 e AST)
SAMPLE_RATE = 16000
N_FFT       = 400       # 25 ms @ 16 kHz
HOP_LENGTH  = 160       # 10 ms @ 16 kHz
N_MELS      = 128
F_MIN       = 0
F_MAX       = 8000      # Nyquist
TOP_DB      = 80.0      # clamping para log-mel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_log_mel(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """
    waveform 16 kHz -> STFT -> mel filterbank -> 128 bands -> log-mel
    Retorna: [n_mels, T_frames]
    """
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        power=2.0,
    ).to(device)

    waveform = waveform.to(device)
    mel_spec = mel_transform(waveform)                # [1, n_mels, T]

    # log-scale com clamping para evitar -inf
    mel_spec = torch.clamp(mel_spec, min=1e-10)
    log_mel  = torch.log10(mel_spec) * 20.0           # dB scale
    log_mel  = torch.clamp(log_mel, min=log_mel.max() - TOP_DB)

    return log_mel.squeeze(0).cpu()                    # [n_mels, T]


def extract_mel_spectrograms(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    skipped = 0
    errors  = 0

    for filepath in tqdm(filelist, desc="Mel-Spectrogram"):
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

        # Stereo -> mono
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        log_mel = compute_log_mel(waveform, sr)
        torch.save(log_mel, output_path)

    total = len(filelist)
    processed = total - skipped - errors
    print(f"\nMel-spectrogram concluido — processados: {processed} | "
          f"pulados (ja existiam): {skipped} | erros: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Extrai log-Mel spectrogram 128 bandas")
    parser.add_argument("-b", "--base-dir",       required=True)
    parser.add_argument("-i", "--input-dir-name", required=True)
    parser.add_argument("-o", "--output-dir-name", default="mel_spectrograms")
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
        if not filelist:
            print(f"Nenhum .wav encontrado em: {input_dir}")
            return

    extract_mel_spectrograms(filelist, input_dir, output_dir)


if __name__ == "__main__":
    main()
