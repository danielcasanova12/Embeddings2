import os
import glob
import argparse
from typing import List, Optional

from os.path import exists, join, relpath, dirname

import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperModel, AutoFeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Modelos suportados
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    "whisper-tiny":      ("openai/whisper-tiny",      39),
    "whisper-base":      ("openai/whisper-base",       74),
    "whisper-small":     ("openai/whisper-small",     244),
    "whisper-medium":    ("openai/whisper-medium",    769),
    "whisper-medium.en": ("openai/whisper-medium.en", 769),
    "whisper-large":     ("openai/whisper-large",    1550),
    "whisper-large-v2":  ("openai/whisper-large-v2", 1550),
    "whisper-large-v3":  ("openai/whisper-large-v3", 1550),
}

# Encoder do Whisper reduz os 3000 frames do mel para 1500 com stride=2
ENCODER_FRAMES_PER_SEC = 50   # 1500 frames / 30 s


def load_model(model_name: str = "whisper-base"):
    """Carrega apenas o encoder do Whisper e o feature extractor."""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Modelo desconhecido: {model_name}. Opções: {list(MODEL_PATHS)}")

    model_path, param_m = MODEL_PATHS[model_name]
    print(f"Carregando {model_name} (~{param_m}M parâmetros) de '{model_path}'...")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = WhisperModel.from_pretrained(model_path).encoder
    model = model.to(device).eval()
    return model, feature_extractor


# ---------------------------------------------------------------------------
# Função principal de extração
# ---------------------------------------------------------------------------

def extract_whisper_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
    model_name: str = "whisper-large-v3",
    layers: Optional[List[int]] = None,
    save_dtype: torch.dtype = torch.float16,
    crop_padding: bool = True,
) -> None:
    """
    Extrai embeddings do encoder Whisper e salva como arquivos .pt.

    Parâmetros
    ----------
    filelist     : lista de caminhos absolutos para os .wav
    input_dir    : diretório raiz dos áudios (usado para calcular caminhos relativos)
    output_dir   : onde salvar os .pt
    model_name   : variante do Whisper (ver MODEL_PATHS)
    layers       : quais camadas salvar. None ou [-1] = só a última.
                   Ex: [-1] → [T, F]
                       [-4, -3, -2, -1] → [4, T, F]
                       None (todas)  → [num_layers, T, F]  ← muito pesado!
    save_dtype   : dtype de gravação. torch.float16 recomendado (metade do espaço).
                   Use torch.float32 se precisar de precisão total.
    crop_padding : se True, remove os frames de silêncio/padding que o Whisper
                   adiciona para completar 30 s. Reduz muito o tamanho para
                   áudios curtos.
    """
    if layers is None:
        layers = [-1]   # padrão seguro: só a última camada

    model, processor = load_model(model_name)
    os.makedirs(output_dir, exist_ok=True)

    skipped = 0
    errors  = 0

    for filepath in tqdm(filelist, desc=f"Whisper [{model_name}]"):
        # ── Verificação de existência ──────────────────────────────────────
        if not exists(filepath):
            print(f"[AVISO] Arquivo não encontrado: {filepath}")
            errors += 1
            continue

        # ── Caminho de saída espelhando a estrutura do input_dir ───────────
        rel_p          = relpath(filepath, input_dir)
        output_path    = join(output_dir, os.path.splitext(rel_p)[0] + ".pt")
        os.makedirs(dirname(output_path), exist_ok=True)

        if exists(output_path):
            skipped += 1
            continue

        # ── Carregar e pré-processar áudio ────────────────────────────────
        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception as e:
            print(f"[ERRO] Falha ao carregar {filepath}: {e}")
            errors += 1
            continue

        # Stereo → mono
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample para 16 kHz se necessário
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        waveform_np = waveform.squeeze().numpy()
        audio_len_sec = waveform_np.shape[-1] / 16000

        # ── Feature extraction (mel spectrogram, padded to 30 s) ──────────
        inputs = processor(
            waveform_np,
            sampling_rate=16000,
            return_tensors="pt",
        )
        model_dtype = next(model.parameters()).dtype
        inputs = {k: v.to(device, dtype=model_dtype) for k, v in inputs.items()}

        # ── Forward pass ──────────────────────────────────────────────────
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple de tensores [1, T, F]

        # ── Selecionar camadas ────────────────────────────────────────────
        num_layers = len(hidden_states)
        resolved   = [i % num_layers for i in layers]   # suporte a índices negativos

        if len(resolved) == 1:
            # Shape final: [T, F]
            embedding = hidden_states[resolved[0]].squeeze(0)
        else:
            # Shape final: [num_selected, T, F]
            embedding = torch.stack([hidden_states[i].squeeze(0) for i in resolved])

        # ── Recortar padding temporal ─────────────────────────────────────
        # Whisper produz sempre 1500 frames (30 s). Para um áudio de 3 s,
        # ~1350 frames são padding puro → descartamos.
        if crop_padding:
            actual_frames = min(
                embedding.shape[-2],                              # T máximo
                max(1, round(audio_len_sec * ENCODER_FRAMES_PER_SEC))
            )
            # Funciona tanto para [T, F] quanto [L, T, F]
            embedding = embedding[..., :actual_frames, :]

        # ── Salvar ────────────────────────────────────────────────────────
        torch.save(embedding.to(save_dtype).cpu(), output_path)

    total = len(filelist)
    processed = total - skipped - errors
    print(
        f"\nWhisper concluído — "
        f"processados: {processed} | pulados (já existiam): {skipped} | erros: {errors}"
    )
    _print_size_tip(layers, save_dtype, crop_padding)


def _print_size_tip(layers, dtype, crop_padding):
    """Imprime estimativa de tamanho baseada nas configurações escolhidas."""
    bytes_per_elem = 2 if dtype == torch.float16 else 4
    n_layers = len(layers) if layers else "todas"
    crop_note = "com crop (tamanho variável)" if crop_padding else "sem crop (T=1500 fixo)"
    print(
        f"  Configuração: {n_layers} camada(s), {bytes_per_elem} bytes/elem, {crop_note}\n"
        f"  Exemplo whisper-large-v3, 5 s, última camada:\n"
        f"    sem otimizações  → ~234 MB\n"
        f"    com todas acima  → ~0.6 MB  (redução de ~390×)"
    )


# ---------------------------------------------------------------------------
# CLI standalone
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrai embeddings do encoder Whisper e salva como .pt"
    )
    parser.add_argument("-b", "--base-dir",       required=True, help="Diretório base do dataset")
    parser.add_argument("-i", "--input-dir-name", required=True, help="Subpasta de áudios dentro do base-dir")
    parser.add_argument("-o", "--output-dir-name", default="whisper_embeddings", help="Subpasta de saída")
    parser.add_argument(
        "-m", "--model-name",
        choices=list(MODEL_PATHS.keys()),
        default="whisper-large-v3",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[-1],
        help=(
            "Índices das camadas a salvar (aceita negativos). "
            "Padrão: -1 (só a última). "
            "Ex: --layers -1  ou  --layers -4 -3 -2 -1"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Dtype de gravação. float16 = metade do espaço. Padrão: float16",
    )
    parser.add_argument(
        "--no-crop-padding",
        action="store_true",
        help="Desativa o recorte de padding temporal (mantém T=1500 fixo)",
    )
    parser.add_argument("-c", "--input-csv",   help="CSV com lista de arquivos (opcional)")
    parser.add_argument("-col", "--column-name", default="filename")

    args = parser.parse_args()

    input_dir  = join(args.base_dir, args.input_dir_name)
    output_dir = join(args.base_dir, args.output_dir_name)

    # Montar filelist
    if args.input_csv and exists(args.input_csv):
        import pandas as pd
        df       = pd.read_csv(args.input_csv)
        filelist = df[args.column_name].tolist()
    else:
        filelist = glob.glob(join(input_dir, "**", "*.wav"), recursive=True)
        if not filelist:
            print(f"Nenhum .wav encontrado em: {input_dir}")
            return

    save_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    extract_whisper_embeddings(
        filelist     = filelist,
        input_dir    = input_dir,
        output_dir   = output_dir,
        model_name   = args.model_name,
        layers       = args.layers,
        save_dtype   = save_dtype,
        crop_padding = not args.no_crop_padding,
    )


if __name__ == "__main__":
    main()