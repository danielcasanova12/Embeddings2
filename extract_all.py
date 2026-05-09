import os
import glob
import argparse
import pandas as pd
from os.path import join, exists

# Importando as funções de extração dos arquivos existentes
from extract_embs.extract_whisper_embeddings import extract_whisper_embeddings
from extract_embs.extract_contentvec import extract_contentvec_embeddings
from extract_embs.extract_speaker_embeddings import extract_speaker_embeddings
from extract_embs.extract_f0 import extract_f0_embeddings

def main():
    parser = argparse.ArgumentParser(description="Extrai todos os embeddings (Whisper, ContentVec, Speaker, F0) e gera/atualiza um CSV.")
    parser.add_argument("-b", "--base-dir", required=True, help="Diretório base do dataset")
    parser.add_argument("-i", "--input-dir-name", required=True, help="Nome da pasta de áudios dentro do base-dir")
    parser.add_argument("-o", "--output-base", default="embeddings", help="Pasta raiz para salvar todos os embeddings (relativo ao base-dir ou absoluto)")
    parser.add_argument("-c", "--csv-path", help="(Opcional) Caminho para o arquivo CSV do dataset existente. Se omitido, buscará todos os wavs e criará um CSV.")
    parser.add_argument("-col", "--column-name", default="filename", help="Se usar -c, coluna do CSV que contém o caminho ou nome do áudio")
    parser.add_argument("--suffix", default="_with_embs.csv", help="Sufixo para salvar o novo arquivo CSV gerado")

    args = parser.parse_args()

    input_dir = join(args.base_dir, args.input_dir_name)
    
    # 1. Carregar lista de arquivos
    filelist = []
    df = None
    
    if args.csv_path and exists(args.csv_path):
        df = pd.read_csv(args.csv_path)
        for f in df[args.column_name]:
            full_path = f if os.path.isabs(str(f)) else join(input_dir, str(f))
            filelist.append(full_path)
    else:
        print(f"Buscando arquivos .wav em: {input_dir}")
        filelist = glob.glob(join(input_dir, "**", "*.wav"), recursive=True)
        if not filelist:
            print("Nenhum arquivo .wav encontrado!")
            return
        df = pd.DataFrame({args.column_name: filelist})

    # 2. Configurar pastas de saída
    output_base = args.output_base if os.path.isabs(args.output_base) else join(args.base_dir, args.output_base)
    output_whisper = join(output_base, "whisper")
    output_content = join(output_base, "contentvec")
    output_speaker = join(output_base, "speaker")
    output_f0      = join(output_base, "f0")

    # 3. Extração em Massa
    print(f"\nTotal de arquivos a processar: {len(filelist)}")
    
    print("\n--- [1/4] Extração: Whisper ---")
    extract_whisper_embeddings(filelist, input_dir, output_whisper, "whisper-large-v3")

    print("\n--- [2/4] Extração: ContentVec ---")
    # pool=True para gerar vetor flat por arquivo [D] para content_vec, ou False se quiser manter temporal [T, D]
    extract_contentvec_embeddings(filelist, input_dir, output_content, "contentvec-best", layer=-1, pool=False)

    print("\n--- [3/4] Extração: Speaker (ECAPA-TDNN) ---")
    # aggregate="mean" gera um vetor unico para o arquivo [D]
    extract_speaker_embeddings(filelist, input_dir, output_speaker, "ecapa-tdnn", aggregate="mean", normalize=True)

    print("\n--- [4/4] Extração: F0 (CREPE) ---")
    extract_f0_embeddings(filelist, input_dir, output_f0, backend="crepe", hop_length=160, quantize=False, n_bins=256, f_min=50, f_max=1100)

    # 4. Atualizar o CSV com os novos caminhos absolutos para facilitar o Dataset
    print("\n--- Atualizando CSV com caminhos dos embeddings ---")
    
    def get_emb_path(audio_path, out_dir, input_dir):
        rel_p = os.path.relpath(audio_path, input_dir)
        emb_file = os.path.splitext(rel_p)[0] + ".pt"
        return join(out_dir, emb_file)

    df["whisper_path"]    = [get_emb_path(f, output_whisper, input_dir) for f in filelist]
    df["contentvec_path"] = [get_emb_path(f, output_content, input_dir) for f in filelist]
    df["speaker_path"]    = [get_emb_path(f, output_speaker, input_dir) for f in filelist]
    df["f0_path"]         = [get_emb_path(f, output_f0, input_dir)      for f in filelist]

    # Salvar o CSV atualizado
    if args.csv_path:
        new_csv_path = args.csv_path.replace(".csv", args.suffix)
    else:
        new_csv_path = join(args.base_dir, "dataset" + args.suffix)
        
    df.to_csv(new_csv_path, index=False)
    
    print(f"\nSucesso! CSV atualizado salvo em: {new_csv_path}")
    print(f"Todos os embeddings salvos em: {output_base}")

if __name__ == "__main__":
    main()