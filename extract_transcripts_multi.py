# extract_transcripts_multi.py
import os
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from faster_whisper import WhisperModel
from transformers import pipeline
from os.path import join, exists

def transcribe_whisper(model_name, audio_paths, device):
    # Forçamos CPU para Whisper ASR para economizar VRAM
    print(f"Carregando Faster-Whisper '{model_name}' na CPU...")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    results = []
    for path in tqdm(audio_paths, desc="Whisper ASR"):
        try:
            segments, info = model.transcribe(path, beam_size=5)
            text = " ".join([segment.text for segment in segments]).strip().lower()
            results.append(text)
        except Exception as e:
            print(f"Erro Whisper em {path}: {e}")
            results.append("")
    return results

def transcribe_wav2vec2(model_id, audio_paths, device):
    # Para Wav2Vec2/pipeline, se o device original for cuda, tentamos manter ou mover para cpu se houver OOM
    # Aqui, para seguir a recomendação, vamos usar CPU (-1 no transformers pipeline)
    print(f"Carregando Wav2Vec2/XLSR '{model_id}' na CPU...")
    asr_pipe = pipeline("automatic-speech-recognition", model=model_id, device=-1)
    results = []
    for path in tqdm(audio_paths, desc="Wav2Vec2 ASR"):
        try:
            res = asr_pipe(path)
            results.append(res["text"].strip().lower())
        except Exception as e:
            print(f"Erro Wav2Vec2 em {path}: {e}")
            results.append("")
    return results

def main():
    parser = argparse.ArgumentParser(description="Transcrição multi-modelo (Whisper + Wav2Vec2/XLSR).")
    parser.add_argument("-i", "--input_csv", required=True)
    parser.add_argument("-col", "--column", default="filename")
    parser.add_argument("-b", "--base_dir", default="")
    parser.add_argument("--whisper_model", default="medium")
    parser.add_argument("--w2v2_model", default="facebook/wav2vec2-large-xlsr-53-english") # Exemplo EN, pode ser alterado por língua
    parser.add_argument("-o", "--output_csv", help="Caminho para o CSV de saída")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    audio_paths = [join(args.base_dir, str(f)) if not os.path.isabs(str(f)) else str(f) for f in df[args.column]]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transcrição com Whisper
    df["transcript_whisper"] = transcribe_whisper(args.whisper_model, audio_paths, device)
    
    # Transcrição com Wav2Vec2
    df["transcript_w2v2"] = transcribe_wav2vec2(args.w2v2_model, audio_paths, device)
    
    output_path = args.output_csv or args.input_csv
    df.to_csv(output_path, index=False)
    print(f"Transcrições salvas em: {output_path}")

if __name__ == "__main__":
    main()
