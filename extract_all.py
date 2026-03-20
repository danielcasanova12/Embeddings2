import os
import argparse
import pandas as pd
from os.path import join, exists

# Importando as funções de extração dos arquivos existentes
from extract_embs.extract_whisper_embeddings import extract_whisper_embeddings
from extract_embs.extract_contentvec import extract_contentvec_embeddings
from extract_embs.extract_speaker_embeddings import extract_speaker_embeddings
from extract_embs.extract_f0 import extract_f0_embeddings

def main():
    parser = argparse.ArgumentParser(description="Extrai todos os embeddings (Whisper, ContentVec, Speaker, F0) e atualiza o CSV.")
    parser.add_argument("-c", "--csv-path", required=True, help="Caminho para o arquivo CSV do dataset (ex: bvcc.csv)")
    parser.add_argument("-b", "--base-dir", required=True, help="Diretório base onde os áudios estão (ou caminho absoluto no CSV)")
    parser.add_argument("-i", "--input-dir-name", required=True, help="Nome da pasta de áudios dentro do base-dir (ou '.' se o CSV já tiver o caminho relativo)")
    parser.add_argument("-o", "--output-base", default="embeddings", help="Pasta raiz para salvar todos os embeddings")
    parser.add_argument("-col", "--column-name", default="filename", help="Coluna do CSV que contém o caminho do áudio")
    
    args = parser.parse_args()

    # 1. Carregar CSV
    if not exists(args.csv_path):
        print(f"Erro: CSV {args.csv_path} não encontrado.")
        return
    
    df = pd.read_csv(args.csv_path)
    # Garantir que os caminhos no CSV sejam absolutos ou relativos ao base_dir/input_dir
    input_dir = join(args.base_dir, args.input_dir_name)
    
    # Lista de arquivos para processar
    filelist = []
    for f in df[args.column_name]:
        full_path = f if os.path.isabs(f) else join(input_dir, f)
        filelist.append(full_path)

    # 2. Configurar pastas de saída
    output_whisper = join(args.output_base, "whisper")
    output_content = join(args.output_base, "contentvec")
    output_speaker = join(args.output_base, "speaker")
    output_f0      = join(args.output_base, "f0")

    # 3. Extração em Massa
    print("--- Iniciando Extração: Whisper ---")
    extract_whisper_embeddings(filelist, input_dir, output_whisper, "whisper-large-v3")

    print("\n--- Iniciando Extração: ContentVec ---")
    extract_contentvec_embeddings(filelist, input_dir, output_content, "contentvec-best", layer=-1, pool=True)

    print("\n--- Iniciando Extração: Speaker (ECAPA-TDNN) ---")
    extract_speaker_embeddings(filelist, input_dir, output_speaker, "ecapa-tdnn", aggregate="mean", normalize=True)

    print("\n--- Iniciando Extração: F0 (CREPE) ---")
    extract_f0_embeddings(filelist, input_dir, output_f0, backend="crepe", hop_length=160, quantize=False, n_bins=256, f_min=50, f_max=1100)

    # 4. Atualizar o CSV com os novos caminhos
    print("\n--- Atualizando CSV com caminhos dos embeddings ---")
    
    def get_emb_path(audio_path, out_dir):
        rel = os.path.relpath(audio_path, input_dir)
        emb_file = os.path.splitext(rel)[0] + ".pt"
        return join(out_dir, emb_file)

    df["whisper_path"]    = [get_emb_path(f, output_whisper) for f in filelist]
    df["contentvec_path"] = [get_emb_path(f, output_content) for f in filelist]
    df["speaker_path"]    = [get_emb_path(f, output_speaker) for f in filelist]
    df["f0_path"]         = [get_emb_path(f, output_f0)      for f in filelist]

    # Salvar o CSV atualizado (sobrescreve ou cria novo)
    new_csv_path = args.csv_path.replace(".csv", "_with_embs.csv")
    df.to_csv(new_csv_path, index=False)
    
    print(f"\nSucesso! CSV atualizado salvo em: {new_csv_path}")
    print(f"Todos os embeddings salvos em: {args.output_base}")

if __name__ == "__main__":
    main()