# scripts/prepare_perturbation_pairs.py
import pandas as pd
import argparse
import os
import random
from tqdm import tqdm

def prepare_pairs(input_csv: str, output_csv: str, emb_types: list, max_pairs=1000):
    """
    Constrói pares para d_quality e d_content usando a coluna transcript.
    """
    df = pd.read_csv(input_csv)
    
    # Filtra transcrições vazias
    df = df[df['transcript'].notna() & (df['transcript'] != "")]
    
    pairs = []
    
    # 1. d_quality: Mesmo transcript, MOS diferentes
    print("Gerando pares d_quality (mesmo conteúdo, qualidades diferentes)...")
    groups = df.groupby('transcript')
    for transcript, group in tqdm(groups):
        if len(group) < 2: continue
        
        # Tenta achar pares com diferença de MOS > 0.5
        sorted_group = group.sort_values('mos')
        for i in range(len(sorted_group)):
            for j in range(i + 1, len(sorted_group)):
                row1 = sorted_group.iloc[i]
                row2 = sorted_group.iloc[j]
                
                if abs(row1['mos'] - row2['mos']) > 0.5:
                    # Adiciona uma entrada por tipo de embedding
                    for emb in emb_types:
                        pairs.append({
                            "pair_type": "quality",
                            "transcript": transcript,
                            "embedding_type": emb,
                            "path_q1": row1[f"{emb}_path"],
                            "path_q2": row2[f"{emb}_path"],
                            "mos_1": row1['mos'],
                            "mos_2": row2['mos'],
                            # Para d_content usaremos placeholders ou deixaremos vazio
                            "path_c1": None,
                            "path_c2": None
                        })
                    if len(pairs) >= max_pairs * len(emb_types) * 2: break
            if len(pairs) >= max_pairs * len(emb_types) * 2: break

    # 2. d_content: Transcripts diferentes, MOS semelhantes (delta < 0.1)
    print("Gerando pares d_content (conteúdos diferentes, mesma qualidade)...")
    # Para d_content, vamos amostrar aleatoriamente pares que atendam ao critério
    df_list = df.to_dict('records')
    count_c = 0
    
    # Amostra limitada para não explodir combinatória
    sample_size = min(len(df_list), 2000)
    sample_df = random.sample(df_list, sample_size)
    
    for i in tqdm(range(len(sample_df))):
        for j in range(i + 1, len(sample_df)):
            row1 = sample_df[i]
            row2 = sample_df[j]
            
            # Critérios: transcripts diferentes, mas qualidade perceptual (MOS) quase igual
            if row1['transcript'] != row2['transcript'] and abs(row1['mos'] - row2['mos']) < 0.1:
                for emb in emb_types:
                    pairs.append({
                        "pair_type": "content",
                        "transcript": f"{row1['transcript']} || {row2['transcript']}",
                        "embedding_type": emb,
                        "path_q1": None,
                        "path_q2": None,
                        "path_c1": row1[f"{emb}_path"],
                        "path_c2": row2[f"{emb}_path"],
                        "mos_1": row1['mos'],
                        "mos_2": row2['mos']
                    })
                count_c += 1
                if count_c >= max_pairs: break
        if count_c >= max_pairs: break

    output_df = pd.DataFrame(pairs)
    output_df.to_csv(output_csv, index=False)
    print(f"Pares salvos em: {output_csv} (Total: {len(output_df)} entradas)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="CSV com transcripts e caminhos de embeddings")
    parser.add_argument("-o", "--output", default="perturbation_pairs.csv")
    parser.add_argument("-e", "--embeddings", nargs='+', default=["whisper", "speaker", "contentvec", "hubert", "wavlm"])
    args = parser.parse_args()
    prepare_pairs(args.input, args.output, args.embeddings)
