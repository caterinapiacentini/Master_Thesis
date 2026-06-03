#!/bin/bash
#SBATCH --job-name=score_gep
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/master_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x-%j.err

set -euo pipefail

# --- UPDATED PATHS ---
DICT_PATH="/home/h12429576/master_thesis/output/DICTIONARY/geoeconomic_dictionary.csv"
TEXT_DIR="/home/h12429576/master_thesis/clean_txt/DATA1"
META_DIR="/home/h12429576/master_thesis/clean_txt/INFO_DATA1"
OUT_DIR="/home/h12429576/master_thesis/output/INDEX"
# ---------------------

ml load python/3.11.11-gcc-15.1.0-u2m6tuq
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/master_thesis/scripts/score_daily_index.py"

$PY -u "$SCRIPT" \
    --dict_path "$DICT_PATH" \
    --text_dir "$TEXT_DIR" \
    --meta_dir "$META_DIR" \
    --output_dir "$OUT_DIR"