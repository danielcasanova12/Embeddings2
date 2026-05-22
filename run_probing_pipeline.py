# run_probing_pipeline.py
import argparse
import subprocess
import os

def run_command(cmd):
    print(f"\n>>> Executando: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Pipeline completo de Linear Probing.")
    parser.add_argument("-i", "--input_csv", required=True, help="CSV inicial (caminhos de áudio e MOS)")
    parser.add_argument("-b", "--base_dir", default="", help="Diretório base para os áudios")
    parser.add_argument("--lang", default="en-us", help="Língua para G2P")
    parser.add_argument("--config", required=True, help="Config YAML do experimento (embeddings e tasks)")
    args = parser.parse_args()

    # 1. Transcrição Multi-ASR
    transcribed_csv = args.input_csv.replace(".csv", "_transcribed.csv")
    run_command([
        "python", "extract_transcripts_multi.py",
        "-i", args.input_csv,
        "-b", args.base_dir,
        "-o", transcribed_csv
    ])

    # 2. Extração de Features Linguísticas
    linguistic_csv = transcribed_csv.replace(".csv", "_linguistic.csv")
    run_command([
        "python", "extract_linguistic_features.py",
        "-i", transcribed_csv,
        "--lang", args.lang,
        "--save_embeddings",
        "-o", linguistic_csv
    ])

    # 3. Rodar Probing
    # Nota: A config YAML deve apontar para o linguistic_csv no campo metadata_path do dataset
    # Ou podemos passar o CSV via override se o sistema suportar
    run_command([
        "python", "scripts/exp5_probing.py",
        "-c", args.config
    ])

    print("\n>>> Pipeline concluído com sucesso!")

if __name__ == "__main__":
    main()
