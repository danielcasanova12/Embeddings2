import os
import glob
import argparse
from typing import List
from os.path import exists, basename, join, relpath, dirname

import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from transformers import Wav2Vec2Model, AutoFeatureExtractor

MODELS = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wav2vec2-base-960h": "facebook/wav2vec2-base-960h",
    "wav2vec2-large": "facebook/wav2vec2-large-960h",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name: str = "wav2vec2-base"):
    if model_name not in MODELS:
        raise ValueError(f"Modelo '{model_name}' não reconhecido. Escolha entre: {list(MODELS.keys())}")
    
    model_path = MODELS[model_name]
    print(f"Carregando modelo: {model_path}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2Model.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model, feature_extractor

def extract_wav2vec2_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
    model_name: str,
    layer: int = -1,
    pool: bool = False,
) -> None:
    model, processor = load_model(model_name)
    os.makedirs(output_dir, exist_ok=True)

    for filepath in tqdm(filelist, desc=f"Extraindo embeddings {model_name}"):
        if not exists(filepath):
            print(f"Arquivo não encontrado: {filepath}")
            continue

        rel_p = relpath(filepath, input_dir)
        output_filename = rel_p.rsplit(".", 1)[0] + ".pt"
        output_filepath = join(output_dir, output_filename)
        os.makedirs(dirname(output_filepath), exist_ok=True)

        if exists(output_filepath):
            continue

        audio_data, sr = torchaudio.load(filepath)

        if audio_data.dim() > 1 and audio_data.shape[0] > 1:
            audio_data = audio_data.mean(dim=0)
        audio_data = audio_data.squeeze()

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio_data = resampler(audio_data)

        inputs = processor(
            audio_data.numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            outputs = model(
                input_values,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states

        if layer == -1:
            embedding = torch.stack(hidden_states).squeeze(1)
        else:
            embedding = hidden_states[layer].squeeze(0)

        if pool:
            embedding = embedding.mean(dim=-2)

        torch.save(embedding.cpu(), output_filepath)

def main():
    parser = argparse.ArgumentParser(description="Extrai embeddings acústicos usando wav2vec 2.0.")
    parser.add_argument("-b", "--base-dir", required=True, help="Caminho para o diretório base")
    parser.add_argument("-i", "--input-dir-name", required=True, help="Nome do diretório de entrada")
    parser.add_argument("-o", "--output-dir-name", default="output_wav2vec2", help="Nome do diretório de saída")
    parser.add_argument("-m", "--model-name", choices=list(MODELS.keys()), default="wav2vec2-base")
    parser.add_argument("-l", "--layer", type=int, default=-1, help="Índice da camada a extrair (-1 para todas)")
    parser.add_argument("--pool", action="store_true", help="Aplica mean-pooling temporal")
    parser.add_argument("-c", "--input-csv", help="Caminho para CSV com lista de arquivos (opcional)")
    parser.add_argument("-col", "--column-name", default="filename", help="Nome da coluna no CSV")

    args = parser.parse_args()

    input_dir  = os.path.join(args.base_dir, args.input_dir_name)
    output_dir = os.path.join(args.base_dir, args.output_dir_name)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        filelist = df[args.column_name].tolist()
    else:
        filelist = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)

    extract_wav2vec2_embeddings(
        filelist=filelist,
        input_dir=input_dir,
        output_dir=output_dir,
        model_name=args.model_name,
        layer=args.layer,
        pool=args.pool,
    )

if __name__ == "__main__":
    main()