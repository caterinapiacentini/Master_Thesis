#!/bin/bash
#SBATCH --job-name=clean_rtrs
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --array=2015-2025%3
#SBATCH --output=/home/%u/master_thesis/logs/%x_%a-%j.out
#SBATCH --error=/home/%u/master_thesis/logs/%x_%a-%j.err

set -euo pipefail

ml purge
ml load python/3.11.11-gcc-15.1.0-u2m6tuq

PY="$HOME/venvs/rtrs_w2v/bin/python"
SCRIPT="$HOME/master_thesis/scripts/clean_rtrs_year_bigrams.py"

YEAR="${SLURM_ARRAY_TASK_ID}"
YEAR_DIR="$HOME/master_thesis/raw_data/RTRS/${YEAR}"

OUTROOT="$HOME/master_thesis/clean_txt"
OUT_DATA="$OUTROOT/DATA"
OUT_INFO="$OUTROOT/INFO_DATA"
LOGDIR="$HOME/master_thesis/logs"

mkdir -p "$OUT_DATA" "$OUT_INFO" "$LOGDIR"

echo "Running on node: $(hostname)"
echo "YEAR=$YEAR"
echo "cpus=$SLURM_CPUS_PER_TASK mem=${SLURM_MEM_PER_NODE:-NA}"

# Count year files
mapfile -t FILES < <(ls -1 "$YEAR_DIR"/*.txt.gz 2>/dev/null | sort)
N=${#FILES[@]}

if [[ "$N" -eq 0 ]]; then
  echo "No .txt.gz files found in $YEAR_DIR"
  exit 0
fi

echo "Found $N files in $YEAR_DIR"

CHUNK=3
PART=0

# Process in chunks of 3 month-files
for ((OFF=0; OFF<N; OFF+=CHUNK)); do
  PART_TAG=$(printf "%03d" "$PART")

  OUTTOK_PART="$OUT_DATA/rtrs_${YEAR}_bigrams_part${PART_TAG}.txt.gz"
  OUTMETA_PART="$OUT_INFO/rtrs_${YEAR}_meta_part${PART_TAG}.jsonl.gz"

  echo "Processing offset=$OFF n_files=$CHUNK -> part=$PART_TAG"

  $PY -u "$SCRIPT" \
    --year_dir "$YEAR_DIR" \
    --out_tokens "$OUTTOK_PART" \
    --out_meta "$OUTMETA_PART" \
    --include_headline \
    --remove_stopwords \
    --drop_diary \
    --min_doc_tokens 10 \
    --min_token_len 2 \
    --phrase_min_count 20 \
    --phrase_threshold 10.0 \
    --file_offset "$OFF" \
    --n_files "$CHUNK" \
    --file_glob "*.txt.gz"

  PART=$((PART+1))
done

# Final yearly outputs
OUTTOK_FINAL="$OUT_DATA/rtrs_${YEAR}_bigrams_nodiary_noboiler.txt.gz"
OUTMETA_FINAL="$OUT_INFO/rtrs_${YEAR}_meta.jsonl.gz"

echo "Concatenating parts ->"
echo "  $OUTTOK_FINAL"
echo "  $OUTMETA_FINAL"

cat "$OUT_DATA"/rtrs_"$YEAR"_bigrams_part*.txt.gz > "$OUTTOK_FINAL"
cat "$OUT_INFO"/rtrs_"$YEAR"_meta_part*.jsonl.gz   > "$OUTMETA_FINAL"

# Remove intermediate parts
rm -f "$OUT_DATA"/rtrs_"$YEAR"_bigrams_part*.txt.gz
rm -f "$OUT_INFO"/rtrs_"$YEAR"_meta_part*.jsonl.gz

echo "Done YEAR=$YEAR"