#!/bin/bash
#SBATCH --job-name=dict_wc_v4
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/master_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x-%j.err

set -euo pipefail

ml purge
ml load python/3.11.11-gcc-15.1.0-u2m6tuq

PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/master_thesis/scripts/make_dict_and_wordcloud.py"

mkdir -p "$HOME/master_thesis/dicts" "$HOME/master_thesis/logs"

$PY -u "$SCRIPT" \
  --kv "$HOME/master_thesis/models/rtrs_1996_2025_w2v.kv" \
  --topn 3000 \
  --min_sim 0.35 \
  --max_terms 100 \
  --min_token_len 3 \
  --max_per_stem 2 \
  --out_parquet "$HOME/master_thesis/dicts/gep_dict_tariff_sanction_top100_v4_dangl.parquet" \
  --out_csv "$HOME/master_thesis/dicts/gep_dict_tariff_sanction_top100_v4_dangl.csv" \
  --out_png "$HOME/master_thesis/dicts/gep_wordcloud_tariff_sanction_top100_v4_dangl.png"
