"""Label/unlabeled splits for SSL."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def stratified_labeled_indices(y: np.ndarray, labeled_per_class: int, rng: np.random.Generator) -> np.ndarray:
    labeled_idx: List[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = rng.permutation(idx)
        labeled_idx.extend(idx[:labeled_per_class].tolist())
    return np.array(labeled_idx)


def split_labeled_unlabeled(y: np.ndarray, labeled_per_class: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    labeled_idx = stratified_labeled_indices(y, labeled_per_class, rng)
    all_idx = np.arange(len(y))
    unlabeled_idx = np.setdiff1d(all_idx, labeled_idx)
    return labeled_idx, unlabeled_idx
