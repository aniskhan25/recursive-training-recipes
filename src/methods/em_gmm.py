"""EM for Gaussian mixture with logging hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class EMHistory:
    means: List[np.ndarray]
    log_likelihood: List[float]
    responsibilities: List[np.ndarray]
    entropy: List[float]


def _gaussian_pdf(x: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    dim = x.shape[1]
    coeff = 1.0 / ((np.sqrt(2 * np.pi) * sigma) ** dim)
    exp = np.exp(-0.5 * np.sum(((x - mean) / sigma) ** 2, axis=1))
    return coeff * exp


def run_em(
    X: np.ndarray,
    init_means: List[List[float]],
    iters: int = 10,
    sigma: float = 1.0,
) -> EMHistory:
    means = [np.array(m, dtype=float) for m in init_means]
    history_means: List[np.ndarray] = []
    history_ll: List[float] = []
    history_resp: List[np.ndarray] = []
    history_entropy: List[float] = []

    for _ in range(iters):
        probs = np.stack([_gaussian_pdf(X, m, sigma) for m in means], axis=1)
        resp = probs / probs.sum(axis=1, keepdims=True)
        history_resp.append(resp)
        history_means.append(np.stack(means))

        ll = np.log(probs.sum(axis=1) + 1e-9).mean()
        history_ll.append(float(ll))

        ent = -np.sum(resp * np.log(resp + 1e-9), axis=1).mean()
        history_entropy.append(float(ent))

        for k in range(len(means)):
            weight = resp[:, k][:, None]
            means[k] = (weight * X).sum(axis=0) / weight.sum()

    return EMHistory(
        means=history_means,
        log_likelihood=history_ll,
        responsibilities=history_resp,
        entropy=history_entropy,
    )
