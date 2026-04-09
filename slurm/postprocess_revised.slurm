#!/bin/bash
#SBATCH --job-name=postproc_8_rev
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/master_thesis/logs/%x-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x-%j.err

set -euo pipefail

PY="$HOME/venvs/rtrs_w2v/bin/python"
INDEX_OUT="$HOME/master_thesis/output/INDEX_8_revised"
THESIS_IDX="$HOME/Master_Thesis/INDEX/index_8_revised"
THESIS_GTM="$HOME/Master_Thesis/GTM_different_versions/GTM8_revised_results"

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

echo "============================================================"
echo "[STEP 1] Copy index CSVs to Master_Thesis repo"
echo "============================================================"
mkdir -p "$THESIS_IDX"
cp "$INDEX_OUT/GEP_Daily_Index.csv"   "$THESIS_IDX/"
cp "$INDEX_OUT/GEP_Monthly_Index.csv" "$THESIS_IDX/"
echo "[OK] CSVs copied"

echo "============================================================"
echo "[STEP 2] Plot monthly index and daily events"
echo "============================================================"
$PY "$THESIS_IDX/plotting_index_8_revised.py"
echo "[OK] Plots generated"

echo "============================================================"
echo "[STEP 3] Compute monthly robustness (day-weighted)"
echo "============================================================"
$PY "$HOME/master_thesis/scripts/scoring/compute_monthly_robustness.py" \
    --daily_path  "$INDEX_OUT/GEP_Daily_Index.csv" \
    --output_dir  "$INDEX_OUT"
cp "$INDEX_OUT/GEP_Monthly_Robustness.csv" "$THESIS_IDX/"
echo "[OK] Robustness CSV computed and copied"

echo "============================================================"
echo "[STEP 4] Robustness check plot"
echo "============================================================"
$PY "$THESIS_IDX/robustness_check_GEP_revised.py" \
    --monthly_path "$THESIS_IDX/GEP_Monthly_Index.csv" \
    --robust_path  "$THESIS_IDX/GEP_Monthly_Robustness.csv" \
    --output_path  "$THESIS_IDX/robustness_check_GEP_revised.png"
echo "[OK] Robustness plot saved"

echo "============================================================"
echo "[STEP 5] Copy GTM results to Master_Thesis repo"
echo "============================================================"
mkdir -p "$THESIS_GTM"
cp "$HOME/master_thesis/output/GTM8_revised_results/"*.csv "$THESIS_GTM/" 2>/dev/null || true
cp "$HOME/master_thesis/output/GTM8_revised_results/"*.txt "$THESIS_GTM/" 2>/dev/null || true
cp "$HOME/master_thesis/output/GTM8_revised_results/"*.png "$THESIS_GTM/" 2>/dev/null || true
cp "$HOME/master_thesis/output/DICTIONARY_8_revised/geoeconomic_dictionary.csv" "$THESIS_GTM/" 2>/dev/null || true
echo "[OK] GTM results copied"

echo "============================================================"
echo "[STEP 6] Git commit"
echo "============================================================"
cd "$HOME/Master_Thesis"
git add INDEX/index_8_revised/ GTM_different_versions/GTM8_revised_results/ slurm/
git commit -m "Add GTM-8 revised index, plots and robustness checks

- Revised seed words for all 8 topics (export controls, sanctions,
  financial coercion, trade war, protectionism improvements)
- GTM8 revised results and geoeconomic dictionary (708 words)
- Daily and monthly GEP index (1996-2025)
- Monthly index plot with annotated peaks
- Daily events plot with 36 geoeconomic events
- Robustness check: article-weighted vs day-weighted aggregation
- Fixed SLURM script path for score_daily_index_new.py

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
echo "[OK] Git commit done"

echo "============================================================"
echo "[ALL DONE] Post-processing complete."
echo "============================================================"
