"""Evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class ClassificationEval:
    loss: float
    accuracy: float


@dataclass
class PseudoLabelEval:
    pseudo_label_fraction: float
    pseudo_label_accuracy: float
    mean_confidence: float
    entropy: float


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return -torch.sum(probs * torch.log(probs + eps), dim=1)


def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> ClassificationEval:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct = 0.0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += float(ce(logits, labels).item())
            pred = logits.argmax(dim=1)
            correct += float((pred == labels).float().sum().item())
            total += labels.numel()
    if total == 0:
        return ClassificationEval(loss=0.0, accuracy=0.0)
    return ClassificationEval(loss=loss_sum / float(total), accuracy=correct / float(total))


def evaluate_pseudo_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> PseudoLabelEval:
    model.eval()
    accepted = 0
    total = 0
    correct_selected = 0.0
    conf_sum = 0.0
    ent_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            probs = torch.softmax(model(images), dim=1)
            conf, pred = torch.max(probs, dim=1)
            mask = conf >= threshold if threshold > 0 else torch.ones_like(conf, dtype=torch.bool)
            if mask.any():
                accepted += int(mask.sum().item())
                correct_selected += float((pred[mask] == labels[mask]).float().sum().item())
            conf_sum += float(conf.sum().item())
            ent_sum += float(_entropy_from_probs(probs).sum().item())
            total += labels.numel()
    if total == 0:
        return PseudoLabelEval(
            pseudo_label_fraction=0.0,
            pseudo_label_accuracy=0.0,
            mean_confidence=0.0,
            entropy=0.0,
        )
    return PseudoLabelEval(
        pseudo_label_fraction=float(accepted) / float(total),
        pseudo_label_accuracy=(correct_selected / float(accepted)) if accepted else 0.0,
        mean_confidence=conf_sum / float(total),
        entropy=ent_sum / float(total),
    )


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    return evaluate_classification(model, loader, device).accuracy
