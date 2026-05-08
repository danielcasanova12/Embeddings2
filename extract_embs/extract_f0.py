import os
import glob
import argparse
from typing import List
from os.path import exists, basename, join, relpath, dirname
 
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
 
 
# ---------------------------------------------------------------------------
# Backends disponíveis para extração de F0
# ---------------------------------------------------------------------------
# "crepe"    → rede neural (torchcrepe) — mais preciso, requer GPU opcional
# "pyin"     → probabilistic YIN via librosa — leve, sem GPU
# "pyworld"  → WORLD vocoder (harvest) — rápido, muito usado em TTS/VC
# "torchyin" → YIN diferenciável (torchyin) — 100% PyTorch
 
 
BACKENDS = ["crepe", "pyin", "pyworld", "torchyin"]
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
# ---------------------------------------------------------------------------
# Quantização de F0 (log-escala uniforme) — "Qf0"
# ---------------------------------------------------------------------------
# Converte Hz contínuo em índice de bin discreto.
# Bins em escala log entre f_min e f_max.
 
def quantize_f0(
    f0: np.ndarray,
    f_min: float = 50.0,
    f_max: float = 1100.0,
    n_bins: int = 256,
) -> np.ndarray:
    """
    Quantiza F0 (Hz) em índices inteiros [0, n_bins].
    Frames não-vozeados (f0 == 0) recebem índice 0 (classe especial).
 
    Retorna array int32 de shape [T].
    """
    bins = np.linspace(np.log(f_min), np.log(f_max), n_bins - 1)
    quantized = np.zeros_like(f0, dtype=np.int32)
    voiced = f0 > 0
    log_f0 = np.log(np.clip(f0[voiced], f_min, f_max))
    quantized[voiced] = np.digitize(log_f0, bins) + 1  # 1-indexed; 0 = não-vozeado
    return quantized
 
 
# ---------------------------------------------------------------------------
# Extração de F0 por backend
# ---------------------------------------------------------------------------
 
def extract_f0_crepe(audio: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Extrai F0 com torchcrepe (neural)."""
    try:
        import torchcrepe
    except ImportError:
        raise ImportError("Instale torchcrepe: pip install torchcrepe")
 
    # Força uma cópia para evitar erro de 'negative strides' no PyTorch
    audio_tensor = torch.from_numpy(audio.copy()).unsqueeze(0).to(device)
    hop_size_sec = hop_length / sr
 
    f0, periodicity = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length=hop_length,
        fmin=50.0,
        fmax=1100.0,
        model="full",
        decoder=torchcrepe.decode.weighted_argmax,
        return_periodicity=True,
        device=device,
    )
    # Aplica threshold de periodicidade para marcar não-vozeados
    f0 = torchcrepe.threshold.At(0.21)(f0, periodicity)
    f0 = f0.squeeze().cpu().numpy()
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0
 
 
def extract_f0_pyin(audio: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Extrai F0 com pYIN (librosa)."""
    try:
        import librosa
    except ImportError:
        raise ImportError("Instale librosa: pip install librosa")
 
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=50.0,
        fmax=1100.0,
        sr=sr,
        hop_length=hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    f0[~voiced_flag] = 0.0
    return f0.astype(np.float32)
 
 
def extract_f0_pyworld(audio: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Extrai F0 com WORLD vocoder (harvest algorithm)."""
    try:
        import pyworld as pw
    except ImportError:
        raise ImportError("Instale pyworld: pip install pyworld")
 
    audio_d = audio.astype(np.float64)
    frame_period_ms = (hop_length / sr) * 1000.0  # hop em milissegundos
    f0, _ = pw.harvest(audio_d, sr, frame_period=frame_period_ms, f0_floor=50.0, f0_ceil=1100.0)
    f0 = f0.astype(np.float32)
    return f0
 
 
def extract_f0_torchyin(audio: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Extrai F0 com torchyin (diferenciável, 100% PyTorch)."""
    try:
        import torchyin
    except ImportError:
        raise ImportError("Instale torchyin: pip install torchyin")
 
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    f0 = torchyin.estimate(audio_tensor, sample_rate=sr, pitch_min=50.0, pitch_max=1100.0)
    f0 = f0.squeeze().numpy()
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float32)
 
 
BACKEND_FN = {
    "crepe":    extract_f0_crepe,
    "pyin":     extract_f0_pyin,
    "pyworld":  extract_f0_pyworld,
    "torchyin": extract_f0_torchyin,
}
 
 
# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
 
def extract_f0_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
    backend: str,
    hop_length: int,
    quantize: bool,
    n_bins: int,
    f_min: float,
    f_max: float,
) -> None:
    """
    Extrai F0 (contínuo ou quantizado) de uma lista de arquivos .wav.
 
    Formatos salvos (arquivo .pt por áudio):
      quantize=False → tensor float32 [T]       — F0 em Hz (0 = não-vozeado)
      quantize=True  → tensor int32  [T]         — índice de bin (0 = não-vozeado)
    """
    extract_fn = BACKEND_FN[backend]
 
    for filepath in tqdm(filelist, desc=f"Extraindo F0 ({backend})"):
        if not exists(filepath):
            print(f"Arquivo não encontrado: {filepath}")
            continue
 
        # Mantém estrutura de subdiretórios
        rel_path      = relpath(filepath, input_dir)
        sub_dir       = dirname(rel_path)
        output_subdir = join(output_dir, sub_dir)
        os.makedirs(output_subdir, exist_ok=True)
 
        # Carrega áudio
        audio_data, sr = torchaudio.load(filepath)
 
        # Estéreo → mono
        if audio_data.dim() > 1 and audio_data.shape[0] > 1:
            audio_data = audio_data.mean(dim=0)
        audio_data = audio_data.squeeze()
 
        # Reamostragem para 16 kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio_data = resampler(audio_data)
            sr = 16000
 
        audio_np = audio_data.numpy()
 
        # Extração de F0
        f0 = extract_fn(audio_np, sr, hop_length)
 
        if quantize:
            result = quantize_f0(f0, f_min=f_min, f_max=f_max, n_bins=n_bins)
            tensor = torch.from_numpy(result)           # int32 [T]
        else:
            tensor = torch.from_numpy(f0)               # float32 [T]
 
        # Salva
        output_filename = basename(filepath).rsplit(".", 1)[0] + ".pt"
        output_filepath = join(output_subdir, output_filename)
        torch.save(tensor, output_filepath)
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Extrai embeddings de F0 (contínuo ou quantizado) de arquivos .wav."
    )
    parser.add_argument(
        "-b", "--base-dir",
        required=True,
        help="Caminho para o diretório base"
    )
    parser.add_argument(
        "-i", "--input-dir-name",
        required=True,
        help="Nome do diretório de entrada (dentro do diretório base)"
    )
    parser.add_argument(
        "-o", "--output-dir-name",
        default="output_f0",
        help="Nome do diretório de saída"
    )
    parser.add_argument(
        "-m", "--backend",
        choices=BACKENDS,
        default="crepe",
        help=(
            "Backend de extração de F0:\n"
            "  crepe    → torchcrepe (neural, mais preciso)\n"
            "  pyin     → probabilistic YIN via librosa\n"
            "  pyworld  → WORLD vocoder harvest\n"
            "  torchyin → YIN diferenciável (puro PyTorch)"
        )
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=160,
        help="Hop length em amostras a 16 kHz (padrão=160 → 10 ms por frame)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Se ativado, quantiza F0 em bins discretos (Qf0)"
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=256,
        help="Número de bins para quantização (padrão=256)"
    )
    parser.add_argument(
        "--f-min",
        type=float,
        default=50.0,
        help="Frequência mínima em Hz para quantização (padrão=50)"
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=1100.0,
        help="Frequência máxima em Hz para quantização (padrão=1100)"
    )
    parser.add_argument(
        "-c", "--input-csv",
        help="Caminho para CSV com lista de arquivos (opcional)"
    )
    parser.add_argument(
        "-col", "--column-name",
        default="filename",
        help="Nome da coluna no CSV que contém os caminhos dos arquivos"
    )
 
    args = parser.parse_args()
 
    input_dir  = os.path.join(args.base_dir, args.input_dir_name)
    output_dir = os.path.join(args.base_dir, args.output_dir_name)
 
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        filelist = df[args.column_name].tolist()
    else:
        filelist = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
 
    os.makedirs(output_dir, exist_ok=True)
 
    print(f"Total de arquivos : {len(filelist)}")
    print(f"Backend           : {args.backend}")
    print(f"Hop length        : {args.hop_length} amostras ({args.hop_length/16:.1f} ms @ 16 kHz)")
    print(f"Quantização (Qf0) : {args.quantize}" + (
        f"  |  bins={args.n_bins}, f_min={args.f_min} Hz, f_max={args.f_max} Hz"
        if args.quantize else ""
    ))
 
    extract_f0_embeddings(
        filelist    = filelist,
        input_dir   = input_dir,
        output_dir  = output_dir,
        backend     = args.backend,
        hop_length  = args.hop_length,
        quantize    = args.quantize,
        n_bins      = args.n_bins,
        f_min       = args.f_min,
        f_max       = args.f_max,
    )
    print("Concluído!")
 
 
if __name__ == "__main__":
    main()