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
#    Rank 1 — highest coverage; distinctive vocabulary, low polysemy
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
#    Rank 2 — escalation framing; dominant in US-China 2018-2020
# ------------------------------------------------------------------
echo "[Processing 2/8: Trade War]"
$PY -u "$SCRIPT" \
    --topic_name   "Trade_War" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    trade_war trade_dispute \
    --pos_weights  1.0 1.0 \
    --neg_words    trade_deal trade_agreement \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 3. Export Controls
#    Rank 3 — critical post-2018 (Huawei, TSMC, semiconductors)
# ------------------------------------------------------------------
echo "[Processing 3/8: Export Controls]"
$PY -u "$SCRIPT" \
    --topic_name   "Export_Controls" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    export_ban export_restriction \
    --pos_weights  1.0 1.0 \
    --neg_words    export_license export_approval \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 4. Tariffs
#    Rank 4 — instrument-level; captures technical articles outside
#    "trade war" framing (Section 301/232, steel/aluminium duties)
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
#    Rank 5 — SWIFT exclusions (Iran 2012, Russia 2022),
#    dollar weaponization, Belt & Road debt leverage
# ------------------------------------------------------------------
echo "[Processing 5/8: Financial Coercion]"
$PY -u "$SCRIPT" \
    --topic_name   "Financial_Coercion" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    financial_exclusion dollar_weaponization \
    --pos_weights  1.0 1.0 \
    --neg_words    financial_integration debt_relief \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 6. Retaliation
#    Rank 6 — reactive escalation language; anchors
#    provocation-retaliation sequences across dyads
# ------------------------------------------------------------------
echo "[Processing 6/8: Retaliation]"
$PY -u "$SCRIPT" \
    --topic_name   "Retaliation" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    retaliatory_tariff countermeasure \
    --pos_weights  1.0 1.0 \
    --neg_words    goodwill_gesture concession \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 7. Protectionism
#    Rank 7 — structural/policy framing; subsidy disputes,
#    industrial policy (Buy American, Made in China 2025)
# ------------------------------------------------------------------
echo "[Processing 7/8: Protectionism]"
$PY -u "$SCRIPT" \
    --topic_name   "Protectionism" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    protectionism import_quota \
    --pos_weights  1.0 1.0 \
    --neg_words    trade_liberalization open_market \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 8. Embargo
#    Rank 8 — lower frequency but semantically tight;
#    Cuba, Iraq/Iran oil, arms embargoes
# ------------------------------------------------------------------
echo "[Processing 8/8: Embargo]"
$PY -u "$SCRIPT" \
    --topic_name   "Embargo" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    trade_embargo arms_embargo \
    --pos_weights  1.0 1.0 \
    --neg_words    embargo_lifted \
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
