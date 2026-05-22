# extract_transcripts.py
import os
import torch
import pandas as pd
import argparse
from tqdm import tqdm
import whisper
from os.path import join, exists

def main():
    parser = argparse.ArgumentParser(description="Extrai transcrições de áudio usando Whisper ASR.")
    parser.add_argument("-i", "--input_csv", required=True, help="CSV com caminhos de áudio")
    parser.add_argument("-col", "--column", default="filename", help="Coluna com o caminho do arquivo")
    parser.add_argument("-b", "--base_dir", default="", help="Diretório base para os áudios")
    parser.add_argument("-m", "--model", default="base", help="Modelo Whisper (tiny, base, small, medium, large)")
    parser.add_argument("-o", "--output_csv", help="Caminho para o CSV de saída")
    args = parser.parse_args()

    print(f"Carregando modelo Whisper '{args.model}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(args.model, device=device)

    df = pd.read_csv(args.input_csv)
    transcripts = []

    print(f"Iniciando transcrição de {len(df)} arquivos...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = row[args.column]
        if not os.path.isabs(audio_path):
            audio_path = join(args.base_dir, audio_path)
        
        if not exists(audio_path):
            print(f"Aviso: Arquivo não encontrado: {audio_path}")
            transcripts.append("")
            continue
            
        try:
            # Realiza a transcrição (apenas texto)
            result = model.transcribe(audio_path, fp16=(device=="cuda"))
            text = result["text"].strip().lower()
            transcripts.append(text)
        except Exception as e:
            print(f"Erro ao processar {audio_path}: {e}")
            transcripts.append("")

    df["transcript"] = transcripts
    
    output_path = args.output_csv or args.input_csv.replace(".csv", "_with_transcripts.csv")
    df.to_csv(output_path, index=False)
    print(f"Transcrições salvas em: {output_path}")

if __name__ == "__main__":
    main()
