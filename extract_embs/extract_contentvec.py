import os
import glob
import argparse
from typing import List
from os.path import exists, basename, join, relpath, dirname

import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from transformers import HubertModel, AutoFeatureExtractor


# ContentVec é baseado na arquitetura HuBERT, mas treinado para separar
# conteúdo (fonética) de informação de locutor.
# Modelos disponíveis no HuggingFace:
#   - "lengyue233/content-vec-best"  (recomendado, 768-dim)
#   - "bshall/hubert-soft"           (versão soft-VC)

CONTENTVEC_MODELS = {
    "contentvec-best": "lengyue233/content-vec-best",
    "hubert-soft":     "bshall/hubert-soft",
    "hubert-base":     "facebook/hubert-base-ls960",
    "hubert-large":    "facebook/hubert-large-ll60k",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str = "contentvec-best"):
    """Carrega o modelo ContentVec / HuBERT e o feature extractor."""
    if model_name not in CONTENTVEC_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' não reconhecido. "
            f"Escolha entre: {list(CONTENTVEC_MODELS.keys())}"
        )
    model_path = CONTENTVEC_MODELS[model_name]
    print(f"Carregando modelo: {model_path}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = HubertModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model, feature_extractor


def extract_contentvec_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
    model_name: str,
    layer: int = -1,        # -1 = todas as camadas; ≥0 = camada específica
    pool: bool = False,     # se True, faz mean-pool no eixo temporal → [F]
) -> None:
    """
    Extrai embeddings de estrutura acústica (ContentVec) de uma lista de arquivos .wav.

    Formato salvo (arquivo .pt por áudio):
      - layer == -1 e pool == False  → [num_layers, T, F]
      - layer == -1 e pool == True   → [num_layers, F]
      - layer >= 0 e pool == False   → [T, F]
      - layer >= 0 e pool == True    → [F]
    """
    model, processor = load_model(model_name)

    for filepath in tqdm(filelist, desc="Extraindo embeddings ContentVec"):
        if not exists(filepath):
            print(f"Arquivo não encontrado: {filepath}")
            continue

        # Mantém estrutura de subdiretórios
        rel_path = relpath(filepath, input_dir)
        sub_dir  = dirname(rel_path)
        output_subdir = join(output_dir, sub_dir)
        os.makedirs(output_subdir, exist_ok=True)

        # Carrega áudio
        audio_data, sr = torchaudio.load(filepath)

        # Estéreo → mono
        if audio_data.dim() > 1 and audio_data.shape[0] > 1:
            audio_data = audio_data.mean(dim=0)
        audio_data = audio_data.squeeze()

        # Reamostragem para 16 kHz (exigido pelo ContentVec/HuBERT)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio_data = resampler(audio_data)

        # Prepara input para o modelo
        inputs = processor(
            audio_data.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            outputs = model(
                input_values,
                output_hidden_states=True,   # retorna todas as camadas ocultas
            )

        # hidden_states: tuple de tensores [1, T, F], uma por camada (incl. embedding)
        hidden_states = outputs.hidden_states  # len = num_layers + 1

        if layer == -1:
            # Empilha todas as camadas → [num_layers, T, F]
            embedding = torch.stack(hidden_states).squeeze(1)
        else:
            # Camada específica → [T, F]
            embedding = hidden_states[layer].squeeze(0)

        if pool:
            embedding = embedding.mean(dim=-2)  # mean-pool no eixo T

        # Salva
        output_filename = basename(filepath).rsplit(".", 1)[0] + ".pt"
        output_filepath = join(output_subdir, output_filename)
        torch.save(embedding.cpu(), output_filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Extrai embeddings de estrutura acústica com ContentVec / HuBERT."
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
        default="output_contentvec",
        help="Nome do diretório de saída"
    )
    parser.add_argument(
        "-m", "--model-name",
        choices=list(CONTENTVEC_MODELS.keys()),
        default="contentvec-best",
        help=(
            "Modelo a usar:\n"
            "  contentvec-best → lengyue233/content-vec-best (recomendado)\n"
            "  hubert-soft     → bshall/hubert-soft\n"
            "  hubert-base     → facebook/hubert-base-ls960\n"
            "  hubert-large    → facebook/hubert-large-ll60k"
        )
    )
    parser.add_argument(
        "-l", "--layer",
        type=int,
        default=-1,
        help=(
            "Índice da camada a extrair (0-indexed incluindo embedding layer). "
            "-1 retorna TODAS as camadas empilhadas. "
            "Para contentvec-best, a camada 9 costuma ser a mais usada."
        )
    )
    parser.add_argument(
        "--pool",
        action="store_true",
        help="Se ativado, aplica mean-pooling no eixo temporal → vetor fixo por arquivo"
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

    # Coleta lista de arquivos
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        filelist = df[args.column_name].tolist()
    else:
        filelist = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Total de arquivos: {len(filelist)}")
    print(f"Camada selecionada: {'todas' if args.layer == -1 else args.layer}")
    print(f"Mean-pool: {args.pool}")

    extract_contentvec_embeddings(
        filelist=filelist,
        input_dir=input_dir,
        output_dir=output_dir,
        model_name=args.model_name,
        layer=args.layer,
        pool=args.pool,
    )
    print("Concluído!")


if __name__ == "__main__":
    main()