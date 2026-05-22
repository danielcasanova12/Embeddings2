# extract_linguistic_features.py
import pandas as pd
import argparse
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Tenta importar phonemizer, se falhar, avisa o usuário
try:
    from phonemizer import phonemize
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False

def extract_phonemes(texts, lang='en-us'):
    if not PHONEMIZER_AVAILABLE:
        print("Aviso: 'phonemizer' não instalado. Pulando extração de fonemas.")
        return [""] * len(texts)
    
    print(f"Convertendo textos para fonemas ({lang})...")
    # Nota: requer espeak-ng instalado no sistema
    try:
        phonemes = phonemize(
            texts,
            language=lang,
            backend='espeak',
            strip=True,
            preserve_punctuation=True,
            njobs=4
        )
        return phonemes
    except Exception as e:
        print(f"Erro no phonemizer: {e}")
        return [""] * len(texts)

def extract_semantic_embeddings(texts, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    print(f"Extraindo embeddings semânticos com {model_name}...")
    model = SentenceTransformer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Extração de representações linguísticas (G2P e Semântica).")
    parser.add_argument("-i", "--input_csv", required=True)
    parser.add_argument("-col", "--column", default="transcript_whisper", help="Coluna de texto para processar")
    parser.add_argument("--lang", default="en-us", help="Língua para G2P (ex: pt-br, en-us, zh)")
    parser.add_argument("--semantic_model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("-o", "--output_csv", help="Caminho para o CSV de saída")
    parser.add_argument("--save_embeddings", action="store_true", help="Salva embeddings semânticos em arquivo .npy separado")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    texts = df[args.column].fillna("").tolist()

    # 1. Fonemas
    df[f"phonemes_{args.column}"] = extract_phonemes(texts, lang=args.lang)

    # 2. Embeddings Semânticos
    sem_embs = extract_semantic_embeddings(texts, model_name=args.semantic_model)
    
    # Salva o caminho do embedding ou apenas confirma no CSV
    if args.save_embeddings:
        emb_path = args.input_csv.replace(".csv", "_semantic_embs.npy")
        import numpy as np
        np.save(emb_path, sem_embs)
        df[f"semantic_emb_path_{args.column}"] = emb_path
        print(f"Embeddings semânticos salvos em: {emb_path}")

    output_path = args.output_csv or args.input_csv.replace(".csv", "_linguistic.csv")
    df.to_csv(output_path, index=False)
    print(f"CSV com features linguísticas salvo em: {output_path}")

if __name__ == "__main__":
    main()
