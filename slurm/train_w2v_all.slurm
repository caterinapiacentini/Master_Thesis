#!/bin/bash
#SBATCH --job-name=w2v_rtrs_all
#SBATCH --partition=hicpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/%u/master_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x-%j.err

set -euo pipefail

ml purge
ml load python/3.11.11-gcc-15.1.0-u2m6tuq

PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/master_thesis/scripts/train_w2v_all.py"

DATA_DIR="$HOME/master_thesis/clean_txt/DATA"
MODEL_DIR="$HOME/master_thesis/models"

$PY -u "$SCRIPT" \
  --data_dir "$DATA_DIR" \
  --out_model "$MODEL_DIR/rtrs_1996_2025_w2v.model" \
  --out_vectors "$MODEL_DIR/rtrs_1996_2025_w2v.kv" \
  --workers 8 \
  --vector_size 300 \
  --window 10 \
  --min_count 20 \
  --negative 10 \
  --epochs 5 \
  --sg 1
