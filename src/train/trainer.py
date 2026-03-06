"""Supervised trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from eval.eval_classification import evaluate_classification
from utils.progress import progress


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
    use_progress: bool = False,
) -> TrainResult:
    model.to(device)
    ce = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []

    for epoch in progress(range(epochs), enabled=use_progress, desc="supervised epochs"):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * labels.numel()
            train_count += labels.numel()

        val = evaluate_classification(model, test_loader, device)
        train_loss = train_loss_sum / float(train_count) if train_count else 0.0

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val.loss,
                "val_accuracy": val.accuracy,
                "test_acc": val.accuracy,
            }
        )

    return TrainResult(history=history)
