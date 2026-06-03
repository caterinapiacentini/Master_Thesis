#!/bin/bash
#SBATCH --job-name=score_gep_8
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/master_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/master_thesis/scripts/score_daily_index_new.py"

DICT_PATH="$HOME/master_thesis/output/DICTIONARY_8/geoeconomic_dictionary.csv"
TEXT_DIR="$HOME/master_thesis/clean_txt/DATA1"
META_DIR="$HOME/master_thesis/clean_txt/INFO_DATA1"
OUT_DIR="$HOME/master_thesis/output/INDEX_8"

FREQ_CAP=4   # max per-article count per dictionary word (Dangl & Salbrechter 2023)
# =============================================================================

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

mkdir -p "$OUT_DIR"
mkdir -p "$HOME/master_thesis/logs"

echo "Building GEP Daily Index (GTM-8 dictionary)"
echo "Dictionary : $DICT_PATH"
echo "Text dir   : $TEXT_DIR"
echo "Meta dir   : $META_DIR"
echo "Output     : $OUT_DIR"
echo "Freq cap   : $FREQ_CAP"
echo "============================================================"

$PY -u "$SCRIPT" \
    --dict_path  "$DICT_PATH"  \
    --text_dir   "$TEXT_DIR"   \
    --meta_dir   "$META_DIR"   \
    --output_dir "$OUT_DIR"    \
    --freq_cap   "$FREQ_CAP"

echo "============================================================"
echo "[OK] GEP Daily Index complete. Results in: $OUT_DIR"
