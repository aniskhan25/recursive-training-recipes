#!/bin/bash
#SBATCH --account=project_2014553
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=nb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --array=0-7

set -euo pipefail

source /scratch/project_2014553/anisrahm/venvs/marimo/bin/activate
export PYTHONPATH=/scratch/project_2014553/anisrahm/recursive-training-recipes/src:${PYTHONPATH:-}

ROOT="/scratch/project_2014553/anisrahm/recursive-training-recipes"
NOTEBOOKS=(
  "01_baseline_supervised"
  "02_state_and_feedback_instrumentation"
  "03_em_gmm_overlap"
  "04_self_training_mnist"
  "05_stability_sweeps"
  "06_fixmatch_cifar10_fewlabels"
  "07_mean_teacher_cifar10_fewlabels"
  "08_hybrid_teacher_threshold"
)

if [ -n "${NB_NAME:-}" ]; then
  NB="$NB_NAME"
else
  NB="${NOTEBOOKS[$SLURM_ARRAY_TASK_ID]}"
fi
NB_IPYNB="$ROOT/notebooks/${NB}.ipynb"
NB_MO="$ROOT/notebooks/${NB}.mo.py"
OUT_HTML="$ROOT/outputs/html/${NB}.html"

mkdir -p "$ROOT/outputs/html" "$ROOT/logs"

marimo convert "$NB_IPYNB" -o "$NB_MO"
marimo export html "$NB_MO" -o "$OUT_HTML"
