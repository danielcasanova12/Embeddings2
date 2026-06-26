# extract_transcripts_multi.py
import os
import sys
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from faster_whisper import WhisperModel
from transformers import pipeline
from os.path import join, exists


def transcribe_whisper(model_name, audio_paths, device):
    # Tenta carregar na GPU, com fallback para CPU
    try:
        compute = "int8_float16" if device == "cuda" else "int8"
        print(f"Carregando Faster-Whisper '{model_name}' em {device} ({compute})...", flush=True)
        model = WhisperModel(model_name, device=device, compute_type=compute)
    except Exception as e:
        print(f"Falha ao carregar em {device} ({e}), usando CPU...", flush=True)
        model = WhisperModel(model_name, device="cpu", compute_type="int8")

    # Modelo CPU de fallback (carregado só se necessário)
    _cpu_model = None

    def _get_cpu_model():
        nonlocal _cpu_model
        if _cpu_model is None:
            print("  Carregando modelo CPU para fallback...", flush=True)
            _cpu_model = WhisperModel(model_name, device="cpu", compute_type="int8")
        return _cpu_model

    results = []
    for path in tqdm(audio_paths, desc="Whisper ASR", file=sys.stderr):
        try:
            segments, info = model.transcribe(path, beam_size=5, language=None)
            text = " ".join([s.text for s in segments]).strip().lower()
            results.append(text)

            lang = getattr(info, "language", "?")
            prob = getattr(info, "language_probability", 0.0)
            print(f"  [{lang} {prob:.2f}] {os.path.basename(path)}: {text[:80]}", flush=True)

        except Exception as e:
            if "libcublas" in str(e) or "cuda" in str(e).lower():
                try:
                    cpu_model = _get_cpu_model()
                    segments, info = cpu_model.transcribe(path, beam_size=5, language=None)
                    text = " ".join([s.text for s in segments]).strip().lower()
                    results.append(text)

                    lang = getattr(info, "language", "?")
                    prob = getattr(info, "language_probability", 0.0)
                    print(f"  [CPU-fallback][{lang} {prob:.2f}] {os.path.basename(path)}: {text[:80]}", flush=True)
                except Exception as e2:
                    print(f"Erro Whisper (CPU fallback) em {path}: {e2}", flush=True)
                    results.append("")
            else:
                print(f"Erro Whisper em {path}: {e}", flush=True)
                results.append("")

    return results


def transcribe_wav2vec2(model_id, audio_paths, device):
    print(f"Carregando Wav2Vec2/XLSR '{model_id}' na CPU...", flush=True)
    asr_pipe = pipeline("automatic-speech-recognition", model=model_id, device=-1)
    results = []
    for path in tqdm(audio_paths, desc="Wav2Vec2 ASR", file=sys.stderr):
        try:
            res = asr_pipe(path)
            text = res["text"].strip().lower()
            results.append(text)
            print(f"  [w2v2] {os.path.basename(path)}: {text[:80]}", flush=True)
        except Exception as e:
            print(f"Erro Wav2Vec2 em {path}: {e}", flush=True)
            results.append("")
    return results


def transcribe_nemo(model_name, audio_paths, device):
    print(f"Carregando NeMo '{model_name}' no {device}...", flush=True)
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    results = []
    for path in tqdm(audio_paths, desc="NeMo ASR", file=sys.stderr):
        try:
            transcriptions = model.transcribe([path], verbose=False)

            if isinstance(transcriptions, tuple):
                hyp = transcriptions[0][0]
            else:
                hyp = transcriptions[0]

            if hasattr(hyp, "text"):
                text = hyp.text
            elif isinstance(hyp, str):
                text = hyp
            else:
                text = str(hyp)

            text = text.strip().lower()
            results.append(text)
            print(f"  [nemo] {os.path.basename(path)}: {text[:80]}", flush=True)

        except Exception as e:
            print(f"Erro NeMo em {path}: {e}", flush=True)
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
    audio_paths = [
        join(args.base_dir, str(f)) if not os.path.isabs(str(f)) else str(f)
        for f in df[args.column]
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.skip_whisper:
        df["transcript_whisper"] = transcribe_whisper(args.whisper_model, audio_paths, device)

    if not args.skip_w2v2:
        df["transcript_w2v2"] = transcribe_wav2vec2(args.w2v2_model, audio_paths, device)

    if not args.skip_nemo:
        df["transcript_nemo"] = transcribe_nemo(args.nemo_model, audio_paths, device)

    output_path = args.output_csv or args.input_csv
    df.to_csv(output_path, index=False)
    print(f"Transcrições salvas em: {output_path}", flush=True)


if __name__ == "__main__":
    main()