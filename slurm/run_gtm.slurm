#!/bin/bash
#SBATCH --job-name=gtm_gep
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/master_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x-%j.err

set -euo pipefail

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

# Navigate to the main directory so the GTM python script 
# creates its output folders in the right place
cd "$HOME/master_thesis"

PY="$HOME/venvs/rtrs_w2v/bin/python"

# --- CORRECTED SCRIPT PATH ---
SCRIPT="$HOME/master_thesis/scripts/run_gtm.py"
# -----------------------------

MODEL_PATH="$HOME/master_thesis/models/new_models/w2v_cbow_64_window_18_50_epochs_1996_2025.pkl"

echo "Starting Guided Topic Modeling..."
echo "Running from: $(pwd)"
echo "Model: $MODEL_PATH"

# Run the Python script with your specific Geoeconomic Pressure seeds
$PY -u "$SCRIPT" \
  --model_path "$MODEL_PATH" \
  --pos_words sanctions tariffs trade_war embargo \
  --pos_weights 1.0 1.0 1.0 1.0 \
  --neg_words free_trade cooperation \
  --neg_weights 1.0 1.0 \
  --size 100 \
  --gravity 1.5

echo "[OK] GTM run completed!"