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
# Modelos disponíveis para extração de Speaker Embeddings
# ---------------------------------------------------------------------------
# "ecapa-tdnn"     → speechbrain/spkrec-ecapa-voxceleb  (recomendado, 192-dim)
# "x-vector"       → speechbrain/spkrec-xvect-voxceleb  (512-dim)
# "resnet-vox"     → speechbrain/spkrec-resnet-voxceleb (256-dim)
# "wespeaker-resnet"→ wespeaker/wespeaker-voxceleb-resnet34-LM (256-dim)
# "wavlm-sv"       → microsoft/wavlm-base-plus-sv       (256-dim, baseado em WavLM)

SPEAKER_MODELS = {
    "ecapa-tdnn":      "speechbrain",   # tratado separadamente
    "x-vector":        "speechbrain",
    "resnet-vox":      "speechbrain",
    "wespeaker-resnet": "wespeaker/wespeaker-voxceleb-resnet34-LM",
    "wavlm-sv":        "microsoft/wavlm-base-plus-sv",
}

SPEECHBRAIN_SOURCES = {
    "ecapa-tdnn": "speechbrain/spkrec-ecapa-voxceleb",
    "x-vector":   "speechbrain/spkrec-xvect-voxceleb",
    "resnet-vox": "speechbrain/spkrec-resnet-voxceleb",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Carregamento de modelo
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    """
    Retorna (model, processor, backend_type).
    backend_type: "speechbrain" | "wespeaker" | "wavlm"
    """
    if model_name not in SPEAKER_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' não reconhecido. "
            f"Escolha entre: {list(SPEAKER_MODELS.keys())}"
        )

    if model_name in SPEECHBRAIN_SOURCES:
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError:
            raise ImportError("Instale speechbrain: pip install speechbrain")
        source = SPEECHBRAIN_SOURCES[model_name]
        print(f"Carregando modelo SpeechBrain: {source}")
        model = EncoderClassifier.from_hparams(
            source=source,
            run_opts={"device": str(device)},
        )
        model.eval()
        return model, None, "speechbrain"

    elif model_name == "wespeaker-resnet":
        try:
            import wespeaker
        except ImportError:
            raise ImportError("Instale wespeaker: pip install wespeaker")
        print(f"Carregando modelo WeSpeaker: {SPEAKER_MODELS[model_name]}")
        model = wespeaker.load_model_local(SPEAKER_MODELS[model_name])
        model.set_device(str(device))
        return model, None, "wespeaker"

    elif model_name == "wavlm-sv":
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector
        except ImportError:
            raise ImportError("Instale transformers: pip install transformers")
        hf_path = SPEAKER_MODELS[model_name]
        print(f"Carregando modelo WavLM-SV: {hf_path}")
        processor = AutoFeatureExtractor.from_pretrained(hf_path)
        model = WavLMForXVector.from_pretrained(hf_path).to(device)
        model.eval()
        return model, processor, "wavlm"


# ---------------------------------------------------------------------------
# Pipeline de extração
# ---------------------------------------------------------------------------

def extract_speaker_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
    model_name: str,
    aggregate: str,    # "mean" | "none"
    normalize: bool,
) -> None:
    """
    Extrai speaker embeddings de uma lista de arquivos .wav.

    Formato salvo (arquivo .pt por áudio):
      aggregate="none" → tensor float32 [D]   — embedding direto do modelo
      aggregate="mean" → tensor float32 [D]   — média de janelas (para áudios longos)

    D varia por modelo:
      ecapa-tdnn       → 192
      x-vector         → 512
      resnet-vox       → 256
      wespeaker-resnet → 256
      wavlm-sv         → 256
    """
    model, processor, backend = load_model(model_name)
    os.makedirs(output_dir, exist_ok=True)

    for filepath in tqdm(filelist, desc=f"Extraindo speaker embeddings ({model_name})"):
        if not exists(filepath):
            print(f"Arquivo não encontrado: {filepath}")
            continue

        # Salva diretamente no output_dir
        rel_p = relpath(filepath, input_dir)
        output_filename = rel_p.rsplit(".", 1)[0] + ".pt"
        output_filepath = join(output_dir, output_filename)
        os.makedirs(dirname(output_filepath), exist_ok=True)

        if exists(output_filepath):
            continue

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

        # ------------------------------------------------------------------
        # Extração por backend
        # ------------------------------------------------------------------
        with torch.no_grad():

            if backend == "speechbrain":
                # SpeechBrain espera [1, T]
                wav = audio_data.unsqueeze(0).to(device)
                if aggregate == "mean":
                    # Divide em janelas de 3 s com overlap de 1 s
                    window   = 3 * sr
                    hop      = 1 * sr
                    chunks   = [wav[:, i:i + window] for i in range(0, wav.shape[1], hop) if i + window // 2 <= wav.shape[1]]
                    
                    # Fallback: Se o áudio for muito curto, usa o áudio inteiro
                    if not chunks:
                        chunks = [wav]

                    embeds   = [model.encode_batch(c).squeeze() for c in chunks]
                    
                    # Garante que embeds seja uma lista de tensores 1D para o mean(dim=0)
                    if len(embeds) == 1 and embeds[0].dim() == 0:
                        embeds = [e.unsqueeze(0) for e in embeds]
                        
                    embedding = torch.stack(embeds).mean(dim=0)
                else:
                    embedding = model.encode_batch(wav).squeeze()  # [D]

            elif backend == "wespeaker":
                audio_np = audio_data.numpy()
                if aggregate == "mean":
                    window   = 3 * sr
                    hop      = 1 * sr
                    embeds   = []
                    for i in range(0, len(audio_np), hop):
                        chunk = audio_np[i:i + window]
                        if len(chunk) < window // 2:
                            break
                        e = model.extract_embedding_from_data(chunk, sr)
                        embeds.append(torch.from_numpy(e))
                    
                    # Fallback: Se a lista estiver vazia por causa de um áudio curto
                    if not embeds:
                        e = model.extract_embedding_from_data(audio_np, sr)
                        embeds.append(torch.from_numpy(e))

                    embedding = torch.stack(embeds).mean(dim=0)
                else:
                    emb = model.extract_embedding_from_data(audio_np, sr)
                    embedding = torch.from_numpy(emb)

            elif backend == "wavlm":
                inputs = processor(
                    audio_data.numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )
                input_values = inputs["input_values"].to(device)
                outputs = model(input_values)
                embedding = outputs.embeddings.squeeze()         # [D]

        # Normalização L2 (útil para similaridade cossenoidal)
        if normalize:
            # Caso a dimensão se perca no squeeze, precisamos garantir que seja 1D
            if embedding.dim() == 0:
                embedding = embedding.unsqueeze(0)
            embedding = torch.nn.functional.normalize(embedding.unsqueeze(0), dim=-1).squeeze(0)

        torch.save(embedding.cpu(), output_filepath)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrai speaker embeddings de arquivos .wav."
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
        default="output_speaker",
        help="Nome do diretório de saída"
    )
    parser.add_argument(
        "-m", "--model-name",
        choices=list(SPEAKER_MODELS.keys()),
        default="ecapa-tdnn",
        help=(
            "Modelo a usar:\n"
            "  ecapa-tdnn       → speechbrain ECAPA-TDNN, 192-dim  (recomendado)\n"
            "  x-vector         → speechbrain x-vector, 512-dim\n"
            "  resnet-vox       → speechbrain ResNet, 256-dim\n"
            "  wespeaker-resnet → WeSpeaker ResNet34, 256-dim\n"
            "  wavlm-sv         → WavLM-Base+ SV, 256-dim"
        )
    )
    parser.add_argument(
        "--aggregate",
        choices=["none", "mean"],
        default="mean",
        help=(
            "Como agregar o embedding no eixo temporal:\n"
            "  mean → média de janelas de 3 s (robusto para áudios longos)\n"
            "  none → passa o áudio inteiro de uma vez"
        )
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Aplica normalização L2 no embedding final (recomendado para similaridade cossenoidal)"
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
    print(f"Modelo            : {args.model_name}")
    print(f"Agregação         : {args.aggregate}")
    print(f"Normalização L2   : {args.normalize}")

    extract_speaker_embeddings(
        filelist   = filelist,
        input_dir  = input_dir,
        output_dir = output_dir,
        model_name = args.model_name,
        aggregate  = args.aggregate,
        normalize  = args.normalize,
    )
    print("Concluído!")


if __name__ == "__main__":
    main()