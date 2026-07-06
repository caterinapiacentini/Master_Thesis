#!/bin/bash
#SBATCH --job-name=gep_val_flags
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --array=1996-2025
#SBATCH --output=/home/%u/Final_Thesis_Clean/logs/%x-%A_%a.out
#SBATCH --error=/home/%u/Final_Thesis_Clean/logs/%x-%A_%a.err

set -euo pipefail

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

# =============================================================================
# CONFIGURATION
# =============================================================================
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/Master_Thesis/code/validation/reconstruct_flags.py"

DICT_CSV="$HOME/Master_Thesis/data/validation/gep_dictionary.csv"
TEXT_DIR="$HOME/Final_Thesis_Clean/clean_txt/DATA_US"
META_DIR="$HOME/Final_Thesis_Clean/clean_txt/INFO_DATA_US"
OUTPUT_DIR="$HOME/Final_Thesis_Clean/output/GEP_Validation/flags"

YEAR="$SLURM_ARRAY_TASK_ID"
# =============================================================================

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/Final_Thesis_Clean/logs"

echo "Reconstructing GEP flags — year $YEAR (min_topics=2, baseline GTM_6 dictionary)"

$PY -u "$SCRIPT" \
    --year       "$YEAR" \
    --dict_csv   "$DICT_CSV" \
    --text_dir   "$TEXT_DIR" \
    --meta_dir   "$META_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --min_topics 2

echo "[OK] Year $YEAR done."
