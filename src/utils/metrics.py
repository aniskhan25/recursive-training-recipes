"""Metrics used across SSL methods."""

from __future__ import annotations

import numpy as np
import torch


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.numel() == 0:
        return 0.0
    return float((y_true == y_pred).float().mean().item())


def acceptance_rate(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def pseudo_label_error(y_true: torch.Tensor, y_pseudo: torch.Tensor) -> float:
    if y_true.numel() == 0:
        return 0.0
    return float((y_true != y_pseudo).float().mean().item())


def teacher_student_disagreement(p_teacher: torch.Tensor, p_student: torch.Tensor) -> float:
    if p_teacher.numel() == 0:
        return 0.0
    return float(torch.mean(torch.abs(p_teacher - p_student)).item())


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    if probs.numel() == 0:
        return 0.0
    confidences, predictions = torch.max(probs, dim=1)
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() > 0:
            acc = (predictions[mask] == labels[mask]).float().mean()
            conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(acc - conf)
    return float(ece.item())


def entropy_from_probs(probs: torch.Tensor) -> float:
    if probs.numel() == 0:
        return 0.0
    eps = 1e-8
    ent = -torch.sum(probs * torch.log(probs + eps), dim=1)
    return float(ent.mean().item())
