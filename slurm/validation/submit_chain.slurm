#!/bin/bash
#SBATCH --job-name=gep_val_chain
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-00:05:00
#SBATCH --output=/home/%u/Final_Thesis_Clean/logs/%x-%j.out
#SBATCH --error=/home/%u/Final_Thesis_Clean/logs/%x-%j.err

# Submits the full GEP validation pipeline as a dependency chain:
#   reconstruct_flags (array, 1996-2025) -> build_sample
# score_annotations.py is NOT chained here: it requires a human to fill in
# human_label first, so run it manually once annotation is done:
#   sbatch --dependency=afterok:<build_sample job> would be pointless since
#   nothing submits it automatically — just run it yourself when ready.

set -euo pipefail

FLAGS_JOB=$(sbatch --parsable \
    "$HOME/Master_Thesis/slurm/validation/reconstruct_flags.slurm")
echo "Submitted reconstruct_flags (array 1996-2025) -> job $FLAGS_JOB"

SAMPLE_JOB=$(sbatch --parsable \
    --dependency=afterok:$FLAGS_JOB \
    "$HOME/Master_Thesis/slurm/validation/build_sample.slurm")
echo "Submitted build_sample -> job $SAMPLE_JOB (dependency: afterok:$FLAGS_JOB)"

echo ""
echo "Chain: $FLAGS_JOB (reconstruct_flags, array) -> $SAMPLE_JOB (build_sample)"
echo "After annotating data/validation/annotation_sample.csv, run manually:"
echo "  \$HOME/venvs/rtrs_w2v/bin/python \$HOME/Master_Thesis/code/validation/score_annotations.py"
