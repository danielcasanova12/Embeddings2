#!/usr/bin/env bash
# =============================================================================
# run_all_extractions.sh
# Extrai embeddings (Whisper, ContentVec, Speaker, F0) para todos os datasets
# e splits (train / val / test).
# =============================================================================
# Adiciona no topo do run_all_extractions.sh, após o shebang
export LD_LIBRARY_PATH=$(python -c "import nvidia.cublas; import os; print(os.path.dirname(nvidia.cublas.__file__))")/lib:$LD_LIBRARY_PATH
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="python"
EXTRACT="$SCRIPT_DIR/extract_all.py"

# Ativa o venv se existir
VENV="$SCRIPT_DIR/venv_embs"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi

# Pasta central de relatórios
REPORT_ROOT="$SCRIPT_DIR/reports"
mkdir -p "$REPORT_ROOT"

# Função auxiliar
run_split() {
    local dataset="$1"
    local split="$2"
    local base_dir="$3"
    local input_dir_name="$4"
    local csv_path="$5"
    local output_base="$6"

    echo ""
    echo "############################################################"
    echo "  DATASET : $dataset"
    echo "  SPLIT   : $split"
    echo "  CSV     : $csv_path"
    echo "  OUTPUT  : $output_base"
    echo "############################################################"

    $PYTHON "$EXTRACT" \
        -b  "$base_dir" \
        -i  "$input_dir_name" \
        -c  "$csv_path" \
        -col "filepath" \
        -o  "$output_base" \
        --report-dir "$REPORT_ROOT/${dataset}_${split}"
}

# =============================================================================
# TMHINTQI
# =============================================================================
TMHI_BASE="/home/time_mos/casanova/Embeddings/datasets/Datasets_mos/TMHINTQI"
TMHI_EMB="/home/time_mos/casanova/Embeddings/embeddings/TMHINTQI"

for split in train val test; do
    run_split "TMHINTQI" "$split" \
        "$TMHI_BASE" \
        "." \
        "$TMHI_BASE/${split}.csv" \
        "$TMHI_EMB/$split"
done

# =============================================================================
# BRSPEECH
# =============================================================================
BRS_BASE="/home/time_mos/casanova/Embeddings/datasets/Datasets_mos/BRSPEECH_MOS_DATASET_v2"
BRS_EMB="/home/time_mos/casanova/Embeddings/embeddings/BRSPEECH"

for split in train val test; do
    run_split "BRSPEECH" "$split" \
        "$BRS_BASE" \
        "." \
        "$BRS_BASE/${split}.csv" \
        "$BRS_EMB/$split"
done

# =============================================================================
# BVCC
# =============================================================================
BVCC_BASE="/home/time_mos/casanova/Embeddings/datasets/Datasets_mos/bvcc/main/DATA"
BVCC_EMB="/home/time_mos/casanova/Embeddings/embeddings/BVCC"

for split in train val test; do
    run_split "BVCC" "$split" \
        "$BVCC_BASE" \
        "." \
        "$BVCC_BASE/sets/${split}.csv" \
        "$BVCC_EMB/$split"
done

# =============================================================================
# SINGMOS
# =============================================================================
SING_BASE="/home/time_mos/casanova/Embeddings/datasets/Datasets_mos/singmos/DATA"
SING_EMB="/home/time_mos/casanova/Embeddings/embeddings/SINGMOS"

for split in train val test; do
    run_split "SINGMOS" "$split" \
        "$SING_BASE" \
        "." \
        "$SING_BASE/sets/${split}.csv" \
        "$SING_EMB/$split"
done

# =============================================================================
# Resumo final dos relatórios
# =============================================================================
echo ""
echo "============================================================"
echo "TODOS OS DATASETS CONCLUÍDOS"
echo "Relatórios JSON em: $REPORT_ROOT"
echo ""
echo "Arquivos de relatório gerados:"
find "$REPORT_ROOT" -name "report_*.json" | sort
echo "============================================================"