#!/bin/bash
#SBATCH --job-name=plot_gep_rob
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/%u/master_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x-%j.err

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/Master_Thesis/INDEX/index_8/robustness_check_GEP.py"

MONTHLY_PATH="$HOME/master_thesis/output/INDEX_8/GEP_Monthly_Index.csv"
ROBUST_PATH="$HOME/master_thesis/output/INDEX_8/GEP_Monthly_Robustness.csv"
OUTPUT_PATH="$HOME/Master_Thesis/INDEX/index_8/robustness_check_GEP.png"

GIT_REPO="$HOME/Master_Thesis"
# =============================================================================

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

echo "Plotting GEP robustness check"
echo "Monthly index : $MONTHLY_PATH"
echo "Robust index  : $ROBUST_PATH"
echo "Output PNG    : $OUTPUT_PATH"
echo "============================================================"

$PY -u "$SCRIPT" \
    --monthly_path "$MONTHLY_PATH" \
    --robust_path  "$ROBUST_PATH"  \
    --output_path  "$OUTPUT_PATH"

echo "============================================================"
echo "[OK] Plot saved. Committing to git..."

cd "$GIT_REPO"
git add "INDEX/index_8/robustness_check_GEP.png"
git commit -m "Add robustness check GEP plot (article-weighted vs day-weighted)"
git push

echo "[OK] Pushed to GitHub. Pull locally to see the plot."
