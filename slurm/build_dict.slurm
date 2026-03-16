#!/bin/bash
#SBATCH --job-name=build_dict
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/master_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x-%j.err

set -euo pipefail

# --- USER PARAMETERS ---
# Modify these to change directories or topic counts without touching Python
IN_DIR="/home/h12429576/master_thesis/output/GTM_6_results"
OUT_DIR="/home/h12429576/master_thesis/output/DICTIONARY"
Q_TOPICS=6
# -----------------------

# Loading modules and identifying the correct python path from your environment
ml load python/3.11.11-gcc-15.1.0-u2m6tuq
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/master_thesis/scripts/build_dictionary.py"

echo "Consolidating GTM results into final dictionary..."

$PY -u "$SCRIPT" \
    --input_dir "$IN_DIR" \
    --output_dir "$OUT_DIR" \
    --num_topics "$Q_TOPICS"

echo "Process completed."