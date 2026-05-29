# extract_transcripts.py
import os
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from faster_whisper import WhisperModel
from os.path import join, exists

def main():
    parser = argparse.ArgumentParser(description="Extrai transcrições de áudio usando Faster-Whisper ASR.")
    parser.add_argument("-i", "--input_csv", required=True, help="CSV com caminhos de áudio")
    parser.add_argument("-col", "--column", default="filename", help="Coluna com o caminho do arquivo")
    parser.add_argument("-b", "--base_dir", default="", help="Diretório base para os áudios")
    parser.add_argument("-m", "--model", default="medium", help="Modelo Faster-Whisper (tiny, base, small, medium, large-v3)")
    parser.add_argument("-o", "--output_csv", help="Caminho para o CSV de saída")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--compute_type", default="int8", help="Compute type (int8, float16, float32)")
    args = parser.parse_args()

    print(f"Carregando modelo Faster-Whisper '{args.model}' no {args.device}...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    df = pd.read_csv(args.input_csv)
    transcripts = []

    print(f"Iniciando transcrição de {len(df)} arquivos...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row[args.column]
        if not os.path.isabs(str(audio_path)):
            audio_path = join(args.base_dir, str(audio_path))
        
        if not exists(audio_path):
            print(f"Aviso: Arquivo não encontrado: {audio_path}")
            transcripts.append("")
            continue
            
        try:
            # Realiza a transcrição (apenas texto)
            segments, info = model.transcribe(audio_path, beam_size=5)
            text = " ".join([segment.text for segment in segments]).strip().lower()
            transcripts.append(text)
        except Exception as e:
            print(f"Erro ao processar {audio_path}: {e}")
            transcripts.append("")

    df["transcript"] = transcripts
    
    output_path = args.output_csv or args.input_csv
    df.to_csv(output_path, index=False)
    print(f"Transcrições salvas em: {output_path}")

if __name__ == "__main__":
    main()
