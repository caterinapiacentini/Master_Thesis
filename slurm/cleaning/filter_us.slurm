#!/bin/bash
#SBATCH --job-name=filter_us
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --array=1996-2025
#SBATCH --output=/home/%u/Final_Thesis_Clean/logs/%x_%a-%j.out
#SBATCH --error=/home/%u/Final_Thesis_Clean/logs/%x_%a-%j.err

set -euo pipefail

ml purge
ml load python/3.11.11-gcc-15.1.0-u2m6tuq

PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/Master_Thesis/code/cleaning/filter_region.py"

YEAR="${SLURM_ARRAY_TASK_ID}"

TOKENS_DIR="$HOME/Final_Thesis_Clean/clean_txt/DATA_WORLD"
META_DIR="$HOME/Final_Thesis_Clean/clean_txt/INFO_DATA_WORLD"
OUT_DIR="$HOME/Final_Thesis_Clean/clean_txt/DATA_US"
OUT_META_DIR="$HOME/Final_Thesis_Clean/clean_txt/INFO_DATA_US"

mkdir -p "$OUT_DIR"
mkdir -p "$OUT_META_DIR"

echo "Filtering year $YEAR to US subset..."

$PY -u "$SCRIPT" \
    --year         "$YEAR" \
    --tokens_dir   "$TOKENS_DIR" \
    --meta_dir     "$META_DIR" \
    --out_dir      "$OUT_DIR" \
    --out_meta_dir "$OUT_META_DIR" \
    --region_tag   "us"

echo "Done year $YEAR"
