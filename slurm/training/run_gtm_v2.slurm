#!/bin/bash
#SBATCH --job-name=gtm6_us_v2_chain
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/%u/Final_Thesis_Clean/logs/%x-%j.out
#SBATCH --error=/home/%u/Final_Thesis_Clean/logs/%x-%j.err

set -euo pipefail

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

# =============================================================================
# CONFIGURATION
# =============================================================================
PY="$HOME/venvs/rtrs_w2v/bin/python"
GTM_SCRIPT="$HOME/Master_Thesis/code/training/gtm/gtm.py"
INDEX_SCRIPT="$HOME/Master_Thesis/code/index/build_index.py"
MODEL_PATH="$HOME/Final_Thesis_Clean/models/us_models/w2v_cbow_64_window_18_100_epochs_us_1996_2025.pkl"

GTM_OUTPUT="$HOME/Final_Thesis_Clean/output/GTM_6_results_US_v2"
INDEX_OUTPUT="$HOME/Final_Thesis_Clean/output/GEP_Index_US_v2"
TEXT_DIR="$HOME/Final_Thesis_Clean/clean_txt/DATA_US"
META_DIR="$HOME/Final_Thesis_Clean/clean_txt/INFO_DATA_US"

CLUSTER_SIZE=100
GRAVITY=1.5
# =============================================================================

cd "$HOME/Final_Thesis_Clean"
mkdir -p "$GTM_OUTPUT/WordClouds"
mkdir -p "$INDEX_OUTPUT"
mkdir -p "$HOME/Final_Thesis_Clean/logs"

echo "================================================================"
echo "STEP 1 — Guided Topic Modeling (US v2, 6 topics, new seeds)"
echo "Model      : $MODEL_PATH"
echo "Output dir : $GTM_OUTPUT"
echo "================================================================"

# ------------------------------------------------------------------
# 1. Sanctions — Ratcheting / maximum-pressure / snap-back angle
#    (vs original "imposition of new sanctions" angle)
# ------------------------------------------------------------------
echo "[1/6: Sanctions — ratcheting / snap-back pressure]"
$PY -u "$GTM_SCRIPT" \
    --topic_name   "Sanctions" \
    --output_dir   "$GTM_OUTPUT" \
    --model_path   "$MODEL_PATH" \
    --pos_words    tougher_sanctions reimpose_sanctions \
    --pos_weights  1.0 1.0 \
    --neg_words    sanctions_relief lifting_sanctions \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 2. Trade_Coercion — Trade-remedies / anti-dumping angle
#    (vs original "tariff war / retaliation" angle)
# ------------------------------------------------------------------
echo "[2/6: Trade_Coercion — anti-dumping / trade remedies]"
$PY -u "$GTM_SCRIPT" \
    --topic_name   "Trade_Coercion" \
    --output_dir   "$GTM_OUTPUT" \
    --model_path   "$MODEL_PATH" \
    --pos_words    countervailing_duties antidumping \
    --pos_weights  1.0 1.0 \
    --neg_words    trade_liberalization trade_pact \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 3. Export_Controls — Semiconductor / chipmaking-equipment denial
#    (vs original "export ban / entity list" that pulls in oil content)
# ------------------------------------------------------------------
echo "[3/6: Export_Controls — semiconductor / chipmaking denial]"
$PY -u "$GTM_SCRIPT" \
    --topic_name   "Export_Controls" \
    --output_dir   "$GTM_OUTPUT" \
    --model_path   "$MODEL_PATH" \
    --pos_words    chipmaking_tools chipmaking_equipment \
    --pos_weights  1.0 1.0 \
    --neg_words    obtain_licenses import_licenses \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 4. Financial_Coercion — Secondary / extraterritorial sanctions
#    (vs original "asset freeze / unilateral sanctions" that near-duplicated Sanctions)
# ------------------------------------------------------------------
echo "[4/6: Financial_Coercion — secondary / extraterritorial pressure]"
$PY -u "$GTM_SCRIPT" \
    --topic_name   "Financial_Coercion" \
    --output_dir   "$GTM_OUTPUT" \
    --model_path   "$MODEL_PATH" \
    --pos_words    extraterritorial_application retaliatory_sanctions \
    --pos_weights  1.0 1.0 \
    --neg_words    debt_forgiveness sanctions_relief \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 5. Embargo — Energy / resource-weaponisation angle
#    (vs original "arms / trade embargo" political-security angle)
# ------------------------------------------------------------------
echo "[5/6: Embargo — energy / resource weaponisation]"
$PY -u "$GTM_SCRIPT" \
    --topic_name   "Embargo" \
    --output_dir   "$GTM_OUTPUT" \
    --model_path   "$MODEL_PATH" \
    --pos_words    trade_embargo weaponisation \
    --pos_weights  1.0 1.0 \
    --neg_words    lift_embargo lifting_sanctions \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# 6. Retaliation — Diplomatic retaliation (visa / travel bans)
#    (vs original "economic retaliation / countermeasures" angle)
# ------------------------------------------------------------------
echo "[6/6: Retaliation — diplomatic / visa-ban retaliation]"
$PY -u "$GTM_SCRIPT" \
    --topic_name   "Retaliation" \
    --output_dir   "$GTM_OUTPUT" \
    --model_path   "$MODEL_PATH" \
    --pos_words    deny_visas visa_bans \
    --pos_weights  1.0 1.0 \
    --neg_words    concessions goodwill_gesture \
    --neg_weights  0.5 0.5 \
    --size         "$CLUSTER_SIZE" \
    --gravity      "$GRAVITY"

# ------------------------------------------------------------------
# Combine all 6 WordClouds into a 2x3 grid
# ------------------------------------------------------------------
echo "[Combining 6 WordClouds into 2x3 grid]"
$PY -u - << 'PYEOF'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

output_dir = os.path.expanduser("~/Final_Thesis_Clean/output/GTM_6_results_US_v2")

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
grid_path = os.path.join(output_dir, "Combined_GEP_Grid_6_US_v2.png")
plt.savefig(grid_path, dpi=300, facecolor="w", edgecolor="w", bbox_inches="tight")
plt.close(fig)
print(f"[OK] Grid saved to {grid_path}")
PYEOF

echo ""
echo "================================================================"
echo "STEP 2 — Index Construction (min_topics=2)"
echo "GTM dir    : $GTM_OUTPUT"
echo "Output dir : $INDEX_OUTPUT"
echo "================================================================"

$PY -u "$INDEX_SCRIPT" \
    --gtm_dir    "$GTM_OUTPUT" \
    --text_dir   "$TEXT_DIR" \
    --meta_dir   "$META_DIR" \
    --output_dir "$INDEX_OUTPUT" \
    --min_topics 2

echo "================================================================"
echo "[OK] Chain complete."
echo "     GTM results : $GTM_OUTPUT"
echo "     Index       : $INDEX_OUTPUT"
echo "================================================================"
