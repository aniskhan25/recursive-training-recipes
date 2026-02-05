# Supplemental_1: PyTorch SSL demos (config-driven)

This folder contains industry-grade, still educational SSL demos aligned to the recursive training tutorial. Every notebook is hypothesis-driven and logs state evolution (pseudo-label accuracy, acceptance rate, calibration proxy, disagreement, and explicit state error where possible).

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

## Notebook order

- 01_baseline_supervised.ipynb
- 02_state_and_feedback_instrumentation.ipynb
- 03_em_gmm_overlap.ipynb
- 04_self_training_mnist.ipynb
- 05_stability_sweeps.ipynb
- 06_fixmatch_cifar10_fewlabels.ipynb
- 07_mean_teacher_cifar10_fewlabels.ipynb
- 08_hybrid_teacher_threshold.ipynb

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
