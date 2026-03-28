#!/bin/bash
#SBATCH --job-name=gtm_gep_8_eur
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
MODEL_PATH="$HOME/master_thesis/models/europe_models/w2v_cbow_64_window_18_50_epochs_europe_1996_2025.pkl"
OUTPUT_DIR="$HOME/master_thesis/output/GTM8_europe_results"

CLUSTER_SIZE=100
GRAVITY=1.5
# =============================================================================

cd "$HOME/master_thesis"

# Create all required output directories before any run
mkdir -p "$OUTPUT_DIR/WordClouds"
mkdir -p "$HOME/master_thesis/logs"

echo "Starting Guided Topic Modeling — 8 Subcategories (Geoeconomic Pressure) [EUROPE model]"
echo "Output directory : $OUTPUT_DIR"
echo "Model            : $MODEL_PATH"
echo "Cluster size     : $CLUSTER_SIZE"
echo "Gravity          : $GRAVITY"
echo "============================================================"

# ------------------------------------------------------------------
# 1. Sanctions
#    Anchors: targeted_sanctions + sanctions_regime
#    [economic_sanctions MISSING in Europe vocab → targeted_sanctions]
#    Negatives: sanctions_relief + sanctions_waiver [both OK]
# ------------------------------------------------------------------
echo "[Processing 1/8: Sanctions]"
$PY -u "$SCRIPT" \
    --topic_name   "Sanctions" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    targeted_sanctions sanctions_regime \
    --pos_weights  1.0 1.0 \
    --neg_words    sanctions_relief sanctions_waiver \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 2. Trade War
#    Anchors: trade_war + trade_friction
#    [trade_conflict MISSING in Europe vocab → trade_friction]
#    Negatives: trade_pact
#    [trade_deal MISSING in Europe vocab → dropped; trade_pact OK]
# ------------------------------------------------------------------
echo "[Processing 2/8: Trade War]"
$PY -u "$SCRIPT" \
    --topic_name   "Trade_War" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    trade_war trade_friction \
    --pos_weights  1.0 1.0 \
    --neg_words    trade_pact \
    --neg_weights  0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 3. Export Controls
#    Anchors: export_ban + export_controls
#    [export_restriction MISSING in Europe vocab → export_controls]
#    Negatives: export_permit
#    [export_license MISSING in Europe vocab → export_permit]
# ------------------------------------------------------------------
echo "[Processing 3/8: Export Controls]"
$PY -u "$SCRIPT" \
    --topic_name   "Export_Controls" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    export_ban export_controls \
    --pos_weights  1.0 1.0 \
    --neg_words    export_permit \
    --neg_weights  0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 4. Tariffs
#    Anchors: import_tariff + customs_duty [both OK]
#    Negatives: tariff_exemption
#    [free_trade MISSING in Europe vocab → dropped; tariff_exemption OK]
# ------------------------------------------------------------------
echo "[Processing 4/8: Tariffs]"
$PY -u "$SCRIPT" \
    --topic_name   "Tariffs" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    import_tariff customs_duty \
    --pos_weights  1.0 1.0 \
    --neg_words    tariff_exemption \
    --neg_weights  0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 5. Financial Coercion
#    Anchors: asset_freeze + frozen_assets [both OK]
#    Negatives: debt_forgiveness
#    [debt_relief MISSING in Europe vocab → debt_forgiveness]
# ------------------------------------------------------------------
echo "[Processing 5/8: Financial Coercion]"
$PY -u "$SCRIPT" \
    --topic_name   "Financial_Coercion" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    asset_freeze frozen_assets \
    --pos_weights  1.0 1.0 \
    --neg_words    debt_forgiveness \
    --neg_weights  0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 6. Retaliation
#    Anchors: retaliation + retaliatory_measures
#    Negatives: concessions + goodwill_gesture
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
#    Negatives: trade_liberalization
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
#    Anchors: trade_embargo + arms_embargo
#    [oil_embargo MISSING in Europe vocab → arms_embargo]
#    Negatives: lift_sanctions
#    [lift_embargo and all variants MISSING in Europe vocab → lift_sanctions]
# ------------------------------------------------------------------
echo "[Processing 8/8: Embargo]"
$PY -u "$SCRIPT" \
    --topic_name   "Embargo" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    trade_embargo arms_embargo \
    --pos_weights  1.0 1.0 \
    --neg_words    lift_sanctions \
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
