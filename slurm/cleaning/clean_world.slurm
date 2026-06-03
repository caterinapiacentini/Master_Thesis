#!/bin/bash
#SBATCH --job-name=clean_world
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G   # Increased memory slightly because the global dataset is much larger
#SBATCH --time=0-23:00:00
#SBATCH --array=1996-2026%5
#SBATCH --output=/home/%u/Final_Thesis_Clean/logs/%x_%a-%j.out
#SBATCH --error=/home/%u/Final_Thesis_Clean/logs/%x_%a-%j.err

set -euo pipefail

ml purge
ml load python/3.11.11-gcc-15.1.0-u2m6tuq

PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/Master_Thesis/code/cleaning/clean_world.py"

YEAR="${SLURM_ARRAY_TASK_ID}"
YEAR_DIR="$HOME/Final_Thesis_Clean/raw_data/${YEAR}"

# --- UPDATED PATHS TO PREVENT OVERWRITING US DATA ---
OUTROOT="$HOME/Final_Thesis_Clean/clean_txt"
OUT_DATA="$OUTROOT/DATA_WORLD"
OUT_INFO="$OUTROOT/INFO_DATA_WORLD"
LOGDIR="$HOME/Final_Thesis_Clean/logs"
# ----------------------------------------------------

mkdir -p "$OUT_DATA" "$OUT_INFO" "$LOGDIR"

echo "Running Global Cleaning for YEAR=$YEAR"

mapfile -t FILES < <(ls -1 "$YEAR_DIR"/*.txt.gz 2>/dev/null | sort)
N=${#FILES[@]}

if [[ "$N" -eq 0 ]]; then
  echo "No files found."
  exit 0
fi

CHUNK=3
PART=0

for ((OFF=0; OFF<N; OFF+=CHUNK)); do
  PART_TAG=$(printf "%03d" "$PART")
  OUTTOK_PART="$OUT_DATA/rtrs_${YEAR}_world_part${PART_TAG}.txt.gz"
  OUTMETA_PART="$OUT_INFO/rtrs_${YEAR}_world_meta_part${PART_TAG}.jsonl.gz"

  $PY -u "$SCRIPT" \
    --year_dir "$YEAR_DIR" \
    --out_tokens "$OUTTOK_PART" \
    --out_meta "$OUTMETA_PART" \
    --include_headline \
    --drop_diary \
    --min_doc_tokens 10 \
    --min_token_len 2

  PART=$((PART+1))
done

# Final outputs for World data
OUTTOK_FINAL="$OUT_DATA/rtrs_${YEAR}_world_nodiary_noboiler.txt.gz"
OUTMETA_FINAL="$OUT_INFO/rtrs_${YEAR}_world_meta.jsonl.gz"

cat "$OUT_DATA"/rtrs_"$YEAR"_world_part*.txt.gz > "$OUTTOK_FINAL"
cat "$OUT_INFO"/rtrs_"$YEAR"_world_meta_part*.jsonl.gz > "$OUTMETA_FINAL"

rm -f "$OUT_DATA"/rtrs_"$YEAR"_world_part*.txt.gz
rm -f "$OUT_INFO"/rtrs_"$YEAR"_world_meta_part*.jsonl.gz

echo "Done Global YEAR=$YEAR"
