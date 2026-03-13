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

cd "$HOME/master_thesis"
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/master_thesis/scripts/run_gtm.py"
MODEL_PATH="$HOME/master_thesis/models/new_models/w2v_cbow_64_window_18_50_epochs_1996_2025.pkl"

echo "Starting Guided Topic Modeling for 6 Subcategories..."

# 1. Trade War
echo "[Processing 1/6: Trade War]"
$PY -u "$SCRIPT" --topic_name "Trade_War" --model_path "$MODEL_PATH" \
  --pos_words trade_war trade_tensions --pos_weights 1.0 1.0 \
  --neg_words free_trade cooperation --neg_weights 0.5 0.5 \
  --size 100 --gravity 1.5

# 2. Tariffs
echo "[Processing 2/6: Tariffs]"
$PY -u "$SCRIPT" --topic_name "Tariffs" --model_path "$MODEL_PATH" \
  --pos_words tariffs punitive_tariffs --pos_weights 1.0 1.0 \
  --neg_words tax_cut --neg_weights 0.5 \
  --size 100 --gravity 1.5

# 3. Sanctions
echo "[Processing 3/6: Sanctions]"
$PY -u "$SCRIPT" --topic_name "Sanctions" --model_path "$MODEL_PATH" \
  --pos_words sanctions targeted_sanctions --pos_weights 1.0 1.0 \
  --neg_words diplomacy --neg_weights 0.5 \
  --size 100 --gravity 1.5

# 4. Embargo
echo "[Processing 4/6: Embargo]"
$PY -u "$SCRIPT" --topic_name "Embargo" --model_path "$MODEL_PATH" \
  --pos_words embargo arms_embargo --pos_weights 1.0 1.0 \
  --neg_words treaty --neg_weights 0.5 \
  --size 100 --gravity 1.5

# 5. Protectionism
echo "[Processing 5/6: Protectionism]"
$PY -u "$SCRIPT" --topic_name "Protectionism" --model_path "$MODEL_PATH" \
  --pos_words protectionism trade_barriers --pos_weights 1.0 1.0 \
  --neg_words open_markets free_trade --neg_weights 0.5 0.5 \
  --size 100 --gravity 1.5

# 6. Retaliation
echo "[Processing 6/6: Retaliation]"
$PY -u "$SCRIPT" --topic_name "Retaliation" --model_path "$MODEL_PATH" \
  --pos_words retaliation retaliatory_measures --pos_weights 1.0 1.0 \
  --neg_words agreement --neg_weights 0.5 \
  --size 100 --gravity 1.5

# 7. Combine into 2x3 Grid
echo "[Combining all images into a 2x3 Grid]"
$PY -u "$SCRIPT" --combine_grid

echo "[OK] All 6 GTM runs and grid generation completed successfully!"