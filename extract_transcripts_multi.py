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
    from faster_whisper import WhisperModel
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

def transcribe_nemo(model_name, audio_paths, device):
    print(f"Carregando NeMo '{model_name}' no {device}...")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    results = []
    # NeMo pode processar em batches, mas para manter consistência com os outros:
    for path in tqdm(audio_paths, desc="NeMo ASR"):
        try:
            transcriptions = model.transcribe([path], verbose=False)
            if isinstance(transcriptions, tuple):
                text = transcriptions[0][0]
            else:
                text = transcriptions[0]
            results.append(text.strip().lower())
        except Exception as e:
            print(f"Erro NeMo em {path}: {e}")
            results.append("")
    return results

def main():
    parser = argparse.ArgumentParser(description="Transcrição multi-modelo (Whisper + Wav2Vec2 + NeMo).")
    parser.add_argument("-i", "--input_csv", required=True)
    parser.add_argument("-col", "--column", default="filename")
    parser.add_argument("-b", "--base_dir", default="")
    parser.add_argument("--whisper_model", default="large-v3")
    parser.add_argument("--w2v2_model", default="facebook/wav2vec2-large-xlsr-53-english") 
    parser.add_argument("--nemo_model", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("-o", "--output_csv", help="Caminho para o CSV de saída")
    parser.add_argument("--skip_whisper", action="store_true")
    parser.add_argument("--skip_w2v2", action="store_true")
    parser.add_argument("--skip_nemo", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    audio_paths = [join(args.base_dir, str(f)) if not os.path.isabs(str(f)) else str(f) for f in df[args.column]]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transcrição com Whisper
    if not args.skip_whisper:
        df["transcript_whisper"] = transcribe_whisper(args.whisper_model, audio_paths, device)
    
    # Transcrição com Wav2Vec2
    if not args.skip_w2v2:
        df["transcript_w2v2"] = transcribe_wav2vec2(args.w2v2_model, audio_paths, device)

    # Transcrição com NeMo
    if not args.skip_nemo:
        df["transcript_nemo"] = transcribe_nemo(args.nemo_model, audio_paths, device)
    
    output_path = args.output_csv or args.input_csv
    df.to_csv(output_path, index=False)
    print(f"Transcrições salvas em: {output_path}")

if __name__ == "__main__":
    main()
