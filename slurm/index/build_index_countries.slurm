#!/bin/bash
#SBATCH --job-name=index_countries
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/Final_Thesis_Clean/logs/%x-%j.out
#SBATCH --error=/home/%u/Final_Thesis_Clean/logs/%x-%j.err

set -euo pipefail

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/Master_Thesis/code/index/build_index_country.py"

GTM_DIR="$HOME/Final_Thesis_Clean/output/GTM_6_results_US"
TEXT_DIR="$HOME/Final_Thesis_Clean/clean_txt/DATA_US"
META_DIR="$HOME/Final_Thesis_Clean/clean_txt/INFO_DATA_US"

mkdir -p "$HOME/Final_Thesis_Clean/logs"

COUNTRIES=(japan uk china iran russia germany)

for COUNTRY in "${COUNTRIES[@]}"; do
    OUTPUT_DIR="$HOME/Final_Thesis_Clean/output/GEP_Index_${COUNTRY^}"
    mkdir -p "$OUTPUT_DIR"
    echo "============================================================"
    echo "Building GEP Index — ${COUNTRY^} (min_topics=2)"
    $PY -u "$SCRIPT" \
        --gtm_dir    "$GTM_DIR" \
        --text_dir   "$TEXT_DIR" \
        --meta_dir   "$META_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --country    "$COUNTRY" \
        --min_topics 2
    echo "[OK] ${COUNTRY^} saved to: $OUTPUT_DIR"
done

echo "============================================================"
echo "[OK] All country indices complete."
