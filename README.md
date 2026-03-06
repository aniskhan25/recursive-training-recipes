# Supplemental_1: PyTorch SSL demos (config-driven)

These notebooks implement minimal educational versions of semi-supervised learning methods.
They prioritize conceptual clarity over exact paper-faithful reproduction.

The goal is to demonstrate recursive training mechanics and stabilization patterns, not to
reproduce full SSL benchmarks or claim SOTA performance.

Every notebook follows the same structure:
1. Concept
2. Why this method exists
3. Algorithm intuition
4. Minimal implementation
5. Experiment
6. Diagnostics
7. Key takeaways
8. Failure modes

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run entrypoint

```bash
python scripts/run.py --config configs/selftrain_mnist.yaml
```

Outputs go to `outputs/logs/` (CSV) and `outputs/figures/` (PNG).

Core logged metrics are standardized across methods:
`train_loss`, `val_loss`, `val_accuracy`, `pseudo_label_fraction`,
`pseudo_label_accuracy`, `mean_confidence`, and `entropy`
(plus method-specific diagnostics like `unsupervised_loss`, `teacher_student_disagreement`, and `ema_gap`).

## Notebook order

- 01_baseline_supervised.ipynb
- 02_state_and_feedback_instrumentation.ipynb
- 03_em_gmm_overlap.ipynb
- 04_self_training_mnist.ipynb
- 05_stability_sweeps.ipynb
- 06_fixmatch_cifar10_fewlabels.ipynb
- 07_mean_teacher_cifar10_fewlabels.ipynb
- 08_hybrid_teacher_threshold.ipynb
- 09_ssl_method_comparison.ipynb

## Appendix: Running notebooks on Puhti (Slurm)

The script `run_notebook_array.sbatch` can run all notebooks or a single one via a job array.

Run all:

```bash
sbatch run_notebook_array.sbatch
```

Run a single notebook:

```bash
sbatch --export=NB_NAME=07_mean_teacher_cifar10_fewlabels run_notebook_array.sbatch
```
