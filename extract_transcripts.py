# extract_transcripts.py
import os
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from faster_whisper import WhisperModel
from os.path import join, exists

def run_transcription(
    input_csv,
    column="filename",
    base_dir="",
    model_name="large-v3",
    output_csv=None,
    device="cuda",
    compute_type="float16",
    beam_size=5,
    checkpoint_every=50
):
    print(f"Carregando modelo Faster-Whisper '{model_name}' no {device}...")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    df = pd.read_csv(input_csv)
    
    # Se a coluna transcript já existe e não está vazia, podemos pular ou perguntar
    # Para simplificar, vamos processar o que estiver faltando ou tudo se a coluna não existir
    if "transcript" not in df.columns:
        df["transcript"] = ""
    
    print(f"Iniciando transcrição de {len(df)} arquivos...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Pula se já houver transcrição (opcional, mas bom para retomar)
        if pd.notna(row.get("transcript")) and str(row.get("transcript")).strip() != "":
            continue

        audio_path = row[column]
        if not os.path.isabs(str(audio_path)):
            audio_path = join(base_dir, str(audio_path))
        
        if not exists(audio_path):
            print(f"Aviso: Arquivo não encontrado: {audio_path}")
            continue
            
        try:
            segments, info = model.transcribe(audio_path, beam_size=beam_size)
            text = " ".join([segment.text for segment in segments]).strip().lower()
            df.at[idx, "transcript"] = text
        except Exception as e:
            print(f"Erro ao processar {audio_path}: {e}")

        # Checkpoint
        if (idx + 1) % checkpoint_every == 0:
            temp_output = output_csv or input_csv
            df.to_csv(temp_output, index=False)

    output_path = output_csv or input_csv
    df.to_csv(output_path, index=False)
    print(f"Transcrições salvas em: {output_path}")
    
    # Limpeza
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Extrai transcrições de áudio usando Faster-Whisper ASR.")
    parser.add_argument("-i", "--input_csv", required=True, help="CSV com caminhos de áudio")
    parser.add_argument("-col", "--column", default="filename", help="Coluna com o caminho do arquivo")
    parser.add_argument("-b", "--base_dir", default="", help="Diretório base para os áudios")
    parser.add_argument("-m", "--model", default="large-v3", help="Modelo Faster-Whisper (tiny, base, small, medium, large-v3)")
    parser.add_argument("-o", "--output_csv", help="Caminho para o CSV de saída")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--compute_type", default="int8", help="Compute type (int8, float16, float32)")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    args = parser.parse_args()

    run_transcription(
        input_csv=args.input_csv,
        column=args.column,
        base_dir=args.base_dir,
        model_name=args.model,
        output_csv=args.output_csv,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        checkpoint_every=args.checkpoint_every
    )

if __name__ == "__main__":
    main()
