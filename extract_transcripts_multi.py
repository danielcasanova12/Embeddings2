# extract_transcripts_multi.py
import os
import torch
import pandas as pd
import argparse
from tqdm import tqdm
import whisper
from transformers import pipeline
from os.path import join, exists

def transcribe_whisper(model_name, audio_paths, device):
    print(f"Carregando Whisper '{model_name}'...")
    model = whisper.load_model(model_name, device=device)
    results = []
    for path in tqdm(audio_paths, desc="Whisper ASR"):
        try:
            res = model.transcribe(path, fp16=(device=="cuda"))
            results.append(res["text"].strip().lower())
        except Exception as e:
            print(f"Erro Whisper em {path}: {e}")
            results.append("")
    return results

def transcribe_wav2vec2(model_id, audio_paths, device):
    print(f"Carregando Wav2Vec2/XLSR '{model_id}'...")
    # Usando pipeline do transformers para facilidade com XLSR
    asr_pipe = pipeline("automatic-speech-recognition", model=model_id, device=0 if device=="cuda" else -1)
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
    audio_paths = [join(args.base_dir, f) if not os.path.isabs(f) else f for f in df[args.column]]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transcrição com Whisper
    df["transcript_whisper"] = transcribe_whisper(args.whisper_model, audio_paths, device)
    
    # Transcrição com Wav2Vec2
    df["transcript_w2v2"] = transcribe_wav2vec2(args.w2v2_model, audio_paths, device)
    
    output_path = args.output_csv or args.input_csv.replace(".csv", "_transcribed.csv")
    df.to_csv(output_path, index=False)
    print(f"Transcrições salvas em: {output_path}")

if __name__ == "__main__":
    main()
