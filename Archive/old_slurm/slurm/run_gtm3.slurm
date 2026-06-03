#!/bin/bash
#SBATCH --job-name=gtm_alt_pos
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/master_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x-%j.err

set -euo pipefail

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

# Create a new specific folder for this robustness check and navigate into it
mkdir -p "$HOME/master_thesis/output_alt_pos"
cd "$HOME/master_thesis/output_alt_pos"

PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/master_thesis/scripts/run_gtm.py"
MODEL_PATH="$HOME/master_thesis/models/new_models/w2v_cbow_64_window_18_50_epochs_1996_2025.pkl"

echo "Starting GTM Robustness Check 2 (Alternative Positives)..."
echo "Running from: $(pwd)"

# Run the Python script with CHANGED positive seeds
$PY -u "$SCRIPT" \
  --model_path "$MODEL_PATH" \
  --pos_words protectionism trade_barriers retaliation boycott \
  --pos_weights 1.0 1.0 1.0 1.0 \
  --neg_words free_trade cooperation \
  --neg_weights 1.0 1.0 \
  --size 100 \
  --gravity 1.5

echo "[OK] GTM Variation 2 completed!"