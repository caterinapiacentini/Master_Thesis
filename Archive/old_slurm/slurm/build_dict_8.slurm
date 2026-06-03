#!/bin/bash
#SBATCH --job-name=build_dict_8
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/master_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/master_thesis/scripts/build_dictionary_8.py"
IN_DIR="$HOME/master_thesis/output/GTM8_results"
OUT_DIR="$HOME/master_thesis/output/DICTIONARY_8"
Q_TOPICS=8
# =============================================================================

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

mkdir -p "$OUT_DIR"
mkdir -p "$HOME/master_thesis/logs"

echo "Consolidating GTM8 results into final geoeconomic pressure dictionary..."
echo "Input  : $IN_DIR"
echo "Output : $OUT_DIR"
echo "Q      : $Q_TOPICS sub-topics"

$PY -u "$SCRIPT" \
    --input_dir  "$IN_DIR"   \
    --output_dir "$OUT_DIR"  \
    --num_topics "$Q_TOPICS"

echo "Process completed."
