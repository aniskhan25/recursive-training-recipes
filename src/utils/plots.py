"""Plotting helpers for logs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=160)


def plot_series(x: Iterable[float], series: Dict[str, Iterable[float]], title: str, ylabel: str) -> None:
    plt.figure(figsize=(5, 3.2))
    for label, ys in series.items():
        plt.plot(list(x), list(ys), marker="o", label=label)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.legend(frameon=False)
