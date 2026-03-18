#!/bin/bash
#SBATCH --job-name=gtm_gep_8
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/master_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x-%j.err

set -euo pipefail

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

# =============================================================================
# CONFIGURATION — all paths and hyperparameters are defined here
# =============================================================================
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/master_thesis/scripts/GTM_8.py"
MODEL_PATH="$HOME/master_thesis/models/new_models/w2v_cbow_64_window_18_50_epochs_1996_2025.pkl"
OUTPUT_DIR="$HOME/master_thesis/output/GTM8_results"

CLUSTER_SIZE=100
GRAVITY=1.5
# =============================================================================

cd "$HOME/master_thesis"

# Create all required output directories before any run
mkdir -p "$OUTPUT_DIR/WordClouds"
mkdir -p "$HOME/master_thesis/logs"

echo "Starting Guided Topic Modeling — 8 Subcategories (Geoeconomic Pressure)"
echo "Output directory : $OUTPUT_DIR"
echo "Model            : $MODEL_PATH"
echo "Cluster size     : $CLUSTER_SIZE"
echo "Gravity          : $GRAVITY"
echo "============================================================"

# ------------------------------------------------------------------
# 1. Sanctions
#    Anchors: economic_sanctions + sanctions_regime
#    Negatives: sanctions_relief + sanctions_waiver [both OK]
# ------------------------------------------------------------------
echo "[Processing 1/8: Sanctions]"
$PY -u "$SCRIPT" \
    --topic_name   "Sanctions" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    economic_sanctions sanctions_regime \
    --pos_weights  1.0 1.0 \
    --neg_words    sanctions_relief sanctions_waiver \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 2. Trade War
#    Anchors: trade_war + trade_conflict
#    Negatives: trade_deal + trade_pact  [trade_agreement MISSING → trade_pact]
# ------------------------------------------------------------------
echo "[Processing 2/8: Trade War]"
$PY -u "$SCRIPT" \
    --topic_name   "Trade_War" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    trade_war trade_conflict \
    --pos_weights  1.0 1.0 \
    --neg_words    trade_deal trade_pact \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 3. Export Controls
#    Anchors: export_ban + export_restriction
#    Negatives: export_license  [export_approval/export_permitted MISSING]
# ------------------------------------------------------------------
echo "[Processing 3/8: Export Controls]"
$PY -u "$SCRIPT" \
    --topic_name   "Export_Controls" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    export_ban export_restriction \
    --pos_weights  1.0 1.0 \
    --neg_words    export_license \
    --neg_weights  0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 4. Tariffs
#    Anchors: import_tariff + customs_duty
#    Negatives: free_trade + tariff_exemption [both OK]
# ------------------------------------------------------------------
echo "[Processing 4/8: Tariffs]"
$PY -u "$SCRIPT" \
    --topic_name   "Tariffs" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    import_tariff customs_duty \
    --pos_weights  1.0 1.0 \
    --neg_words    free_trade tariff_exemption \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 5. Financial Coercion
#    Anchors: asset_freeze + frozen_assets
#    [All "financial_*", "dollar_weaponization", "swift_exclusion" MISSING —
#    these concepts are too recent/niche for the model's vocabulary.
#    asset_freeze + frozen_assets are the best confirmed anchors in this space]
#    Negatives: debt_relief  [debt_trap MISSING]
# ------------------------------------------------------------------
echo "[Processing 5/8: Financial Coercion]"
$PY -u "$SCRIPT" \
    --topic_name   "Financial_Coercion" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    asset_freeze frozen_assets \
    --pos_weights  1.0 1.0 \
    --neg_words    debt_relief \
    --neg_weights  0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 6. Retaliation
#    Anchors: retaliation + retaliatory_measures
#    [Previous run used retaliatory_tariff + countermeasure — too far
#    apart in the embedding space, caused random drift with angles >1.0]
#    Negatives: concessions + goodwill_gesture [both OK]
# ------------------------------------------------------------------
echo "[Processing 6/8: Retaliation]"
$PY -u "$SCRIPT" \
    --topic_name   "Retaliation" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    retaliation retaliatory_measures \
    --pos_weights  1.0 1.0 \
    --neg_words    concessions goodwill_gesture \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 7. Protectionism
#    Anchors: protectionism + import_quota
#    Negatives: trade_liberalization  [open_market/open_markets/free_markets MISSING]
# ------------------------------------------------------------------
echo "[Processing 7/8: Protectionism]"
$PY -u "$SCRIPT" \
    --topic_name   "Protectionism" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    protectionism import_quota \
    --pos_weights  1.0 1.0 \
    --neg_words    trade_liberalization \
    --neg_weights  0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 8. Embargo
#    Anchors: trade_embargo + oil_embargo
#    [Swapped arms_embargo → oil_embargo: arms_embargo was already
#    captured at word #52 in the Sanctions run, so using oil_embargo
#    pulls this topic toward energy/commodity coercion instead]
#    Negatives: lift_embargo  [embargo_lifted/lifting_embargo MISSING]
# ------------------------------------------------------------------
echo "[Processing 8/8: Embargo]"
$PY -u "$SCRIPT" \
    --topic_name   "Embargo" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    trade_embargo oil_embargo \
    --pos_weights  1.0 1.0 \
    --neg_words    lift_embargo \
    --neg_weights  0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# Combine all 8 WordClouds into a 2x4 grid
# ------------------------------------------------------------------
echo "[Combining all 8 images into a 2x4 grid]"
$PY -u "$SCRIPT" \
    --combine_grid \
    --output_dir "$OUTPUT_DIR"

echo "============================================================"
echo "[OK] All 8 GTM runs and grid generation completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
