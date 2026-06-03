#!/bin/bash
#SBATCH --job-name=gtm_new_final
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/Final_Thesis_Clean/logs/%x-%j.out
#SBATCH --error=/home/%u/Final_Thesis_Clean/logs/%x-%j.err

set -euo pipefail

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

# =============================================================================
# CONFIGURATION
# =============================================================================
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/Master_Thesis/code/training/gtm/gtm.py"
MODEL_PATH="$HOME/Final_Thesis_Clean/models/new_models/w2v_cbow_64_window_18_50_epochs_1996_2025.pkl"
OUTPUT_DIR="$HOME/Final_Thesis_Clean/output/GTM_new_final_results"

CLUSTER_SIZE=100
GRAVITY=1.5
# =============================================================================

cd "$HOME/Final_Thesis_Clean"

mkdir -p "$OUTPUT_DIR/WordClouds"
mkdir -p "$HOME/Final_Thesis_Clean/logs"

echo "Starting Guided Topic Modeling — GEP New Final (6 topics)"
echo "Output directory : $OUTPUT_DIR"
echo "Model            : $MODEL_PATH"
echo "Cluster size     : $CLUSTER_SIZE  |  Gravity: $GRAVITY"
echo "============================================================"

# ------------------------------------------------------------------
# 1. Sanctions
#    Target: comprehensive economic isolation regimes used as GEP tools
#    Events: Iran nuclear, Russia-Ukraine, North Korea, Iraq
#    Pos: economic_sanctions (the instrument) + targeted_sanctions (modern smart approach)
#    Neg: sanctions_relief + sanctions_waiver (cooperative resolution, not coercive use)
# ------------------------------------------------------------------
echo "[Processing 1/6: Sanctions]"
$PY -u "$SCRIPT" \
    --topic_name   "Sanctions" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    economic_sanctions targeted_sanctions \
    --pos_weights  1.0 1.0 \
    --neg_words    sanctions_relief sanctions_waiver \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 2. Trade_Coercion
#    Target: tariff wars and trade weaponisation as geopolitical tools
#    Events: US-China 2018-2019, Liberation Day 2025, steel/aluminium tariffs
#    Pos: trade_war (the conflict frame) + retaliatory_tariffs (the coercive instrument)
#    Neg: trade_deal + trade_pact (cooperative resolution)
#    Note: using retaliatory_tariffs instead of trade_conflict avoids market-sentiment drift
# ------------------------------------------------------------------
echo "[Processing 2/6: Trade_Coercion]"
$PY -u "$SCRIPT" \
    --topic_name   "Trade_Coercion" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    trade_war retaliatory_tariffs \
    --pos_weights  1.0 1.0 \
    --neg_words    trade_deal trade_pact \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 3. Export_Controls
#    Target: technology and goods access denial as strategic GEP weapon
#    Events: Huawei 2019, Russia tech exports 2022, CHIPS Act
#    Pos: export_ban (the restriction) + entity_list (the US policy instrument)
#    Neg: export_license + export_licenses (permissive, not restrictive)
#    Note: entity_list anchors toward US national-security-driven tech denial
# ------------------------------------------------------------------
echo "[Processing 3/6: Export_Controls]"
$PY -u "$SCRIPT" \
    --topic_name   "Export_Controls" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    export_ban entity_list \
    --pos_weights  1.0 1.0 \
    --neg_words    export_license export_licenses \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 4. Financial_Coercion
#    Target: weaponisation of the dollar and financial system
#    Events: Russia 2022 reserve freeze, SWIFT exclusion, Iran banking sanctions,
#            oligarch asset seizures
#    Pos: asset_freeze (direct seizure) + secondary_sanctions (dollar-coercion of 3rd parties)
#    Neg: debt_relief (cooperative financial assistance, not punitive)
#    Note: secondary_sanctions anchors toward the US extraterritorial financial pressure tool
# ------------------------------------------------------------------
echo "[Processing 4/6: Financial_Coercion]"
$PY -u "$SCRIPT" \
    --topic_name   "Financial_Coercion" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    asset_freeze secondary_sanctions \
    --pos_weights  1.0 1.0 \
    --neg_words    debt_relief \
    --neg_weights  0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 5. Embargo
#    Target: sectoral / commodity trade bans as geopolitical pressure
#    Events: Cuba embargo, Iraq oil-for-food, Iran oil embargo, Russia energy
#    Pos: trade_embargo + oil_embargo (two anchor points: broad + energy-specific)
#    Neg: lift_embargo + lifting_sanctions (end-state resolution)
# ------------------------------------------------------------------
echo "[Processing 5/6: Embargo]"
$PY -u "$SCRIPT" \
    --topic_name   "Embargo" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    trade_embargo oil_embargo \
    --pos_weights  1.0 1.0 \
    --neg_words    lift_embargo lifting_sanctions \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 6. Retaliation
#    Target: escalatory tit-for-tat responses across all GEP instruments
#    Events: US-China counter-tariffs, Russia countermeasures, Canada/EU retaliation
#    Pos: retaliation + countermeasures (escalation mechanism)
#    Neg: concessions + goodwill_gesture (de-escalation / cooperative response)
# ------------------------------------------------------------------
echo "[Processing 6/6: Retaliation]"
$PY -u "$SCRIPT" \
    --topic_name   "Retaliation" \
    --output_dir   "$OUTPUT_DIR" \
    --model_path   "$MODEL_PATH" \
    --pos_words    retaliation countermeasures \
    --pos_weights  1.0 1.0 \
    --neg_words    concessions goodwill_gesture \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# Combine all 6 WordClouds into a 2x3 grid
# ------------------------------------------------------------------
echo "[Combining all 6 topic WordClouds into a 2x3 grid]"
$PY -u - << 'PYEOF'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

output_dir = os.path.expanduser("~/Final_Thesis_Clean/output/GTM_new_final_results")

image_files = [
    "topic_Sanctions.png",
    "topic_Trade_Coercion.png",
    "topic_Export_Controls.png",
    "topic_Financial_Coercion.png",
    "topic_Embargo.png",
    "topic_Retaliation.png",
]

titles = [
    "Sanctions",
    "Trade Coercion",
    "Export Controls",
    "Financial Coercion",
    "Embargo",
    "Retaliation",
]

fig, axes = plt.subplots(2, 3, figsize=(30, 14))
axes = axes.flatten()

for idx, (img_name, title) in enumerate(zip(image_files, titles)):
    img_path = os.path.join(output_dir, "WordClouds", img_name)
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=28, pad=16)
    else:
        print(f"[WARNING] Missing: {img_path}")
    axes[idx].axis("off")

plt.tight_layout()
grid_path = os.path.join(output_dir, "Combined_GEP_Grid_6.png")
plt.savefig(grid_path, dpi=300, facecolor='w', edgecolor='w', bbox_inches='tight')
plt.close(fig)
print(f"[OK] Grid saved to {grid_path}")
PYEOF

echo "============================================================"
echo "[OK] All 6 GTM runs completed. Results in: $OUTPUT_DIR"
