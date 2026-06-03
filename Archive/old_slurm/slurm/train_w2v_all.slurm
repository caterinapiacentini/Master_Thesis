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

# --- MODIFIED DIRECTORIES ---
DATA_DIR="$HOME/master_thesis/clean_txt/DATA1"
MODEL_DIR="$HOME/master_thesis/models/new_models"

# Ensure the new models directory exists
mkdir -p "$MODEL_DIR"
# ----------------------------

# --- UPDATED PARAMETERS FOR GTM ---
$PY -u "$SCRIPT" \
  --data_dir "$DATA_DIR" \
  --out_model "$MODEL_DIR/w2v_cbow_64_window_18_100_epochs_1996_2025.model" \
  --out_pkl "$MODEL_DIR/w2v_cbow_64_window_18_100_epochs_1996_2025.pkl" \
  --workers 24 \
  --vector_size 64 \
  --window 18 \
  --min_count 20 \
  --negative 10 \
  --epochs 100 \
  --sg 0