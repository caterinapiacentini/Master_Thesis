#!/bin/bash
#SBATCH --job-name=w2v_rtrs_us
#SBATCH --partition=hicpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=14-00:00:00
#SBATCH --output=/home/%u/Final_Thesis_Clean/logs/%x-%j.out
#SBATCH --error=/home/%u/Final_Thesis_Clean/logs/%x-%j.err

set -euo pipefail

ml purge
ml load python/3.11.11-gcc-15.1.0-u2m6tuq

PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/Master_Thesis/code/training/word2vec/train_w2v.py"

DATA_DIR="$HOME/Final_Thesis_Clean/clean_txt/DATA_US"
MODEL_DIR="$HOME/Final_Thesis_Clean/models/us_models"

mkdir -p "$MODEL_DIR"

$PY -u "$SCRIPT" \
  --data_dir    "$DATA_DIR" \
  --out_model   "$MODEL_DIR/w2v_cbow_64_window_18_100_epochs_us_1996_2025.model" \
  --out_pkl     "$MODEL_DIR/w2v_cbow_64_window_18_100_epochs_us_1996_2025.pkl" \
  --workers     64 \
  --vector_size 64 \
  --window      18 \
  --min_count   20 \
  --negative    10 \
  --epochs      100 \
  --sg          0
