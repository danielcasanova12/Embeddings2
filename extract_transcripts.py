"""
extract_transcripts.py
======================
Extrai transcrições ASR de arquivos de áudio com suporte a:
  - openai-whisper  (padrão, roda em CPU ou CUDA)
  - faster-whisper  (--backend faster, recomendado para CUDA com pouca VRAM)
  - transformers    (--backend transformers, usa SDPA para GPU com alta velocidade)

Funcionalidades extras:
  - Retoma de onde parou: pula linhas que já têm 'transcript' preenchido.
  - Checkpoint periódico: salva o CSV a cada N arquivos.
  - Limpeza de VRAM ao encerrar.
  - Arquivo de saída configurável (padrão: <input>_with_transcripts.csv).

Exemplos de uso
---------------
# openai-whisper, modelo medium, CPU
python extract_transcripts.py -i data.csv -m medium

# faster-whisper, medium, CUDA com int8 (ideal p/ GPU com pouca VRAM)
python extract_transcripts.py -i data.csv -m medium --backend faster --compute-type int8_float16

# transformers, medium, CUDA (Aceleração nativa PyTorch SDPA)
python extract_transcripts.py -i data.csv -m medium --backend transformers --device cuda
"""

import os
import gc
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from faster_whisper import WhisperModel
from os.path import join, exists


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _transcribe_openai(model, audio_path: str, use_fp16: bool) -> str:
    """Transcreve com openai-whisper."""
    result = model.transcribe(audio_path, fp16=use_fp16)
    return result["text"].strip().lower()


def _transcribe_faster(model, audio_path: str) -> str:
    """Transcreve com faster-whisper."""
    segments, _ = model.transcribe(audio_path, beam_size=5)
    return " ".join(seg.text for seg in segments).strip().lower()


# ---------------------------------------------------------------------------
# Loader de modelos
# ---------------------------------------------------------------------------

def load_openai_whisper(model_name: str, device: str):
    import whisper  # noqa: WPS433
    print(f"[openai-whisper] Carregando '{model_name}' em {device.upper()}...")
    return whisper.load_model(model_name, device=device)


def load_faster_whisper(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel  # noqa: WPS433
    except ImportError:
        raise ImportError(
            "faster-whisper não instalado. Execute:\n"
            "  pip install faster-whisper"
        )
    print(
        f"[faster-whisper] Carregando '{model_name}' em {device.upper()} "
        f"com compute_type='{compute_type}'..."
    )
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def load_transformers_whisper(model_name: str, device: str):
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError:
        raise ImportError("Execute: pip install transformers accelerate")

    # Define precisão (float16) para economizar VRAM na GPU
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[transformers] Carregando '{model_name}' em {device.upper()} via SDPA (Aceleração Nativa)...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        f"openai/whisper-{model_name}",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa"  # Aceleração nativa sem precisar compilar C++
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(f"openai/whisper-{model_name}")

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if device == "cuda" else -1,
        chunk_length_s=30  # Proteção essencial contra OOM em áudios longos
    )


# ---------------------------------------------------------------------------
# Núcleo da extração
# ---------------------------------------------------------------------------

def run_transcription(
    input_csv: str,
    column: str,
    base_dir: str,
    model_name: str,
    output_csv: str,
    backend: str,
    device: str,
    compute_type: str,
    checkpoint_every: int,
):
    df = pd.read_csv(input_csv)

    # Garante que a coluna existe
    if "transcript" not in df.columns:
        df["transcript"] = None

    # Identifica apenas linhas sem transcrição
    missing_mask = df["transcript"].isna() | (df["transcript"].astype(str).str.strip() == "")
    missing_indices = df[missing_mask].index.tolist()

    if not missing_indices:
        print("Coluna 'transcript' já preenchida para todos os registros. Nada a fazer.")
        df.to_csv(output_csv, index=False)
        print(f"CSV salvo em: {output_csv}")
        return

    total = len(df)
    pending = len(missing_indices)
    print(f"Total de registros : {total}")
    print(f"Já transcritos     : {total - pending}")
    print(f"A transcrever      : {pending}")

    # Carrega o modelo escolhido
    if backend == "faster":
        model = load_faster_whisper(model_name, device, compute_type)
        transcribe_fn = lambda path: _transcribe_faster(model, path)  # noqa: E731
    elif backend == "transformers":
        pipe = load_transformers_whisper(model_name, device)
        transcribe_fn = lambda path: pipe(path)["text"].strip().lower()  # noqa: E731
    else:
        use_fp16 = device == "cuda"
        model = load_openai_whisper(model_name, device)
        transcribe_fn = lambda path: _transcribe_openai(model, path, use_fp16)  # noqa: E731

    errors = 0
    try:
        for i, idx in enumerate(tqdm(missing_indices, desc="Transcrevendo"), start=1):
            raw_path = df.at[idx, column]
            audio_path = raw_path if os.path.isabs(str(raw_path)) else join(base_dir, str(raw_path))

            if not exists(audio_path):
                print(f"\n[aviso] Arquivo não encontrado: {audio_path}")
                df.at[idx, "transcript"] = ""
                errors += 1
                continue

            try:
                df.at[idx, "transcript"] = transcribe_fn(audio_path)
            except Exception as exc:
                print(f"\n[erro] {audio_path}: {exc}")
                df.at[idx, "transcript"] = ""
                errors += 1

            # Checkpoint periódico
            if i % checkpoint_every == 0:
                df.to_csv(output_csv, index=False)
                print(f"\n  [checkpoint] {i}/{pending} transcrições salvas → {output_csv}")

    finally:
        # Salva progresso mesmo em caso de interrupção
        df.to_csv(output_csv, index=False)
        clear_vram()

    print(f"\nTranscrições salvas em : {output_csv}")
    print(f"Erros/ausências        : {errors}/{pending}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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

    run_transcription(
        input_csv        = args.input_csv,
        column           = args.column,
        base_dir         = args.base_dir,
        model_name       = args.model,
        output_csv       = output_csv,
        backend          = args.backend,
        device           = args.device,
        compute_type     = args.compute_type,
        checkpoint_every = args.checkpoint_every,
    )

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