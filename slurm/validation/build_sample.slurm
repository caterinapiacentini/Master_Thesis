#!/bin/bash
#SBATCH --job-name=gep_val_sample
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/%u/Final_Thesis_Clean/logs/%x-%j.out
#SBATCH --error=/home/%u/Final_Thesis_Clean/logs/%x-%j.err

set -euo pipefail

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

# =============================================================================
# CONFIGURATION
# =============================================================================
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/Master_Thesis/code/validation/build_sample.py"

FLAGS_DIR="$HOME/Final_Thesis_Clean/output/GEP_Validation/flags"
RAW_DATA_DIR="$HOME/Final_Thesis_Clean/raw_data"
OUTPUT="$HOME/Master_Thesis/data/validation/annotation_sample.csv"
# =============================================================================

mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "$HOME/Final_Thesis_Clean/logs"

echo "Building GEP validation annotation sample"
echo "Flags dir  : $FLAGS_DIR"
echo "Raw data   : $RAW_DATA_DIR"
echo "Output     : $OUTPUT"
echo "============================================================"

$PY -u "$SCRIPT" \
    --flags_dir    "$FLAGS_DIR" \
    --raw_data_dir "$RAW_DATA_DIR" \
    --output       "$OUTPUT"

echo "============================================================"
echo "[OK] Annotation sample saved to: $OUTPUT"
