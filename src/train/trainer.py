"""Supervised trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainResult:
    history: List[Dict[str, float]]


def run_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> TrainResult:
    model.to(device)
    ce = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.numel()

        history.append({"epoch": float(epoch), "test_acc": correct / total if total else 0.0})

    return TrainResult(history=history)
