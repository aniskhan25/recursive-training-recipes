"""Synthetic GMM data with overlap control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SyntheticGMM:
    X: np.ndarray
    y: np.ndarray


def make_overlapping_gmm(
    rng: np.random.Generator,
    n_samples: int,
    means: List[List[float]],
    cov_scale: float,
    overlap: float,
) -> SyntheticGMM:
    n_components = len(means)
    dim = len(means[0])
    per = n_samples // n_components
    Xs = []
    ys = []
    cov = cov_scale * np.eye(dim)
    for idx, mean in enumerate(means):
        mean = np.array(mean) * (1.0 - overlap)
        X = rng.multivariate_normal(mean, cov, size=per)
        y = np.full(per, idx)
        Xs.append(X)
        ys.append(y)
    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    perm = rng.permutation(len(X_all))
    return SyntheticGMM(X=X_all[perm], y=y_all[perm])
