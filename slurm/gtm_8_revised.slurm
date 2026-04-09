#!/bin/bash
#SBATCH --job-name=gtm_gep_8_rev
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
# CONFIGURATION
# =============================================================================
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/master_thesis/scripts/GTM_8.py"
MODEL_PATH="$HOME/master_thesis/models/new_models/w2v_cbow_64_window_18_50_epochs_1996_2025.pkl"
OUTPUT_DIR="$HOME/master_thesis/output/GTM8_revised_results"

CLUSTER_SIZE=100
GRAVITY=1.5
# =============================================================================

cd "$HOME/master_thesis"

mkdir -p "$OUTPUT_DIR/WordClouds"
mkdir -p "$HOME/master_thesis/logs"

echo "Starting GTM — 8 Revised Subcategories (Geoeconomic Pressure)"
echo "Output directory : $OUTPUT_DIR"
echo "Model            : $MODEL_PATH"
echo "Cluster size     : $CLUSTER_SIZE"
echo "Gravity          : $GRAVITY"
echo "============================================================"
echo "Revision rationale:"
echo "  Sanctions       : broadened beyond Iran-nuclear (sectoral + targeted)"
echo "  Trade War       : trade_frictions replaces trade_conflict; financial_markets as neg"
echo "  Export Controls : entity_list + export_controls replace energy-drifted seeds"
echo "  Financial Coerc : frozen_russian + payment_freeze; IMF/debt as explicit negatives"
echo "  Protectionism   : economic_nationalism added to differentiate from Tariffs"
echo "============================================================"

# ------------------------------------------------------------------
# 1. Sanctions
#    Revision: broadened from Iran-nuclear to multi-regime sanctions.
#    Old seeds (economic_sanctions + sanctions_regime) pulled entirely
#    toward Iran nuclear deal coverage. New seeds (sectoral_sanctions +
#    targeted_sanctions) are not tied to any single country/issue and
#    cover the full sanctions landscape (Russia, NK, Venezuela, etc.).
# ------------------------------------------------------------------
echo "[Processing 1/8: Sanctions]"
$PY -u "$SCRIPT" \
    --topic_name   "Sanctions" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    sectoral_sanctions targeted_sanctions \
    --pos_weights  1.0 1.0 \
    --neg_words    sanctions_relief sanctions_waiver \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 2. Trade War
#    Revision: trade_frictions replaces trade_conflict (less ambiguous
#    in embedding space); financial_markets added as negative to push
#    away from the market-reaction language (stocks, tumble, wall_street)
#    that contaminated the original topic.
# ------------------------------------------------------------------
echo "[Processing 2/8: Trade War]"
$PY -u "$SCRIPT" \
    --topic_name   "Trade_War" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    trade_war trade_frictions \
    --pos_weights  1.0 1.0 \
    --neg_words    trade_deal trade_pact financial_markets \
    --neg_weights  0.5 0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 3. Export Controls
#    Revision: original seeds (export_ban + export_restriction) drifted
#    into domestic US energy/EPA regulation (biofuel waivers, ethanol
#    mandates). New seeds (entity_list + export_controls) specifically
#    anchor to technology export control policy — the Huawei ban,
#    semiconductor restrictions, and entity list blacklists.
# ------------------------------------------------------------------
echo "[Processing 3/8: Export Controls]"
$PY -u "$SCRIPT" \
    --topic_name   "Export_Controls" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    entity_list export_controls \
    --pos_weights  1.0 1.0 \
    --neg_words    export_license export_permit \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 4. Tariffs
#    Unchanged: import_tariff + customs_duty are well-anchored and the
#    topic is clean. Kept as-is for comparability with original GTM-8.
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
#    Revision: original seeds (asset_freeze + frozen_assets) drifted
#    toward IMF/multilateral debt restructuring (imf, paris_club, hipc).
#    New seeds (frozen_russian + payment_freeze) specifically anchor to
#    sanctions-driven financial coercion — freezing of sovereign assets,
#    SWIFT exclusion, payment blockades. IMF/debt terms are explicit
#    negatives to prevent re-drift.
# ------------------------------------------------------------------
echo "[Processing 5/8: Financial Coercion]"
$PY -u "$SCRIPT" \
    --topic_name   "Financial_Coercion" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    frozen_russian payment_freeze \
    --pos_weights  1.0 1.0 \
    --neg_words    debt_restructuring imf_program \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 6. Retaliation
#    Unchanged: retaliation + retaliatory_measures are well-anchored
#    and the topic is clean.
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
#    Revision: economic_nationalism replaces import_quota as second
#    anchor. import_quota overlapped heavily with the Tariffs topic.
#    economic_nationalism captures the ideological/political dimension
#    of protectionism (industrial policy, domestic preference, subsidy
#    regimes) and differentiates this topic from instrument-focused Tariffs.
# ------------------------------------------------------------------
echo "[Processing 7/8: Protectionism]"
$PY -u "$SCRIPT" \
    --topic_name   "Protectionism" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    protectionism economic_nationalism \
    --pos_weights  1.0 1.0 \
    --neg_words    trade_liberalization \
    --neg_weights  0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 8. Embargo
#    Unchanged: trade_embargo + oil_embargo are well-anchored.
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
echo "[OK] All 8 revised GTM runs completed. Results in: $OUTPUT_DIR"
