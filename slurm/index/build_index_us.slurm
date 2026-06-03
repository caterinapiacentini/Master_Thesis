#!/bin/bash
#SBATCH --job-name=index_us
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/%u/Final_Thesis_Clean/logs/%x-%j.out
#SBATCH --error=/home/%u/Final_Thesis_Clean/logs/%x-%j.err

set -euo pipefail

ml load python/3.11.11-gcc-15.1.0-u2m6tuq

# =============================================================================
# CONFIGURATION
# =============================================================================
PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/Master_Thesis/code/index/build_index.py"

GTM_DIR="$HOME/Final_Thesis_Clean/output/GTM_6_results_US"
TEXT_DIR="$HOME/Final_Thesis_Clean/clean_txt/DATA_US"
META_DIR="$HOME/Final_Thesis_Clean/clean_txt/INFO_DATA_US"
OUTPUT_DIR="$HOME/Final_Thesis_Clean/output/GEP_Index_US"
# =============================================================================

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HOME/Final_Thesis_Clean/logs"

echo "Building GEP Index — US (GTM_6 US results, min_topics=2)"
echo "GTM dir    : $GTM_DIR"
echo "Text dir   : $TEXT_DIR"
echo "Meta dir   : $META_DIR"
echo "Output dir : $OUTPUT_DIR"
echo "============================================================"

echo "[Step 1/1: Official index (min_topics=2)]"
$PY -u "$SCRIPT" \
    --gtm_dir    "$GTM_DIR" \
    --text_dir   "$TEXT_DIR" \
    --meta_dir   "$META_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --min_topics 2

echo "============================================================"
echo "[OK] US GEP index (min_topics=2) saved to: $OUTPUT_DIR"
