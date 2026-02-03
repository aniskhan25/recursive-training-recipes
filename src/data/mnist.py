"""MNIST data loaders for SSL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms

from data.splits import split_labeled_unlabeled
from data.augment import mnist_weak, mnist_strong


@dataclass
class SSLDataLoaders:
    labeled: DataLoader
    unlabeled: DataLoader
    test: DataLoader
    unlabeled_eval: DataLoader


def get_mnist_ssl(
    data_dir: str,
    labeled_per_class: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> SSLDataLoaders:
    root = Path(data_dir)
    rng = np.random.default_rng(seed)

    base = MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    y = np.array(base.targets)
    labeled_idx, unlabeled_idx = split_labeled_unlabeled(y, labeled_per_class, rng)

    labeled_ds = MNIST(root, train=True, download=True, transform=mnist_weak())
    unlabeled_ds = MNIST(root, train=True, download=True, transform=mnist_strong())

    labeled = DataLoader(Subset(labeled_ds, labeled_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    unlabeled = DataLoader(Subset(unlabeled_ds, unlabeled_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_ds = MNIST(root, train=False, download=True, transform=transforms.ToTensor())
    test = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    unlabeled_eval_ds = MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    unlabeled_eval = DataLoader(Subset(unlabeled_eval_ds, unlabeled_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return SSLDataLoaders(labeled=labeled, unlabeled=unlabeled, test=test, unlabeled_eval=unlabeled_eval)
