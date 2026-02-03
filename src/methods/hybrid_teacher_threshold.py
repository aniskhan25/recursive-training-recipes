"""Hybrid teacher + threshold method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import acceptance_rate, accuracy, teacher_student_disagreement


@dataclass
class HybridResult:
    history: List[Dict[str, float]]


def _update_ema(teacher: nn.Module, student: nn.Module, ema_decay: float) -> None:
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data = ema_decay * t_param.data + (1.0 - ema_decay) * s_param.data


def run_hybrid(
    student: nn.Module,
    teacher: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    unlabeled_eval: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    ema_decay: float,
    tau: float,
    lambda_u: float,
) -> HybridResult:
    student.to(device)
    teacher.to(device)
    ce = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        student.train()
        labeled_iter = iter(labeled_loader)
        for (u_images, _), _ in zip(unlabeled_loader, range(len(unlabeled_loader))):
            try:
                l_images, l_labels = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                l_images, l_labels = next(labeled_iter)

            l_images, l_labels = l_images.to(device), l_labels.to(device)
            u_images = u_images.to(device)

            logits_l = student(l_images)
            loss_sup = ce(logits_l, l_labels)

            with torch.no_grad():
                t_probs = torch.softmax(teacher(u_images), dim=1)
                conf, pseudo = torch.max(t_probs, dim=1)
                mask = conf >= tau

            s_logits = student(u_images)
            loss_unsup = ce(s_logits[mask], pseudo[mask]) if mask.any() else torch.tensor(0.0, device=device)

            loss = loss_sup + lambda_u * loss_unsup
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _update_ema(teacher, student, ema_decay)

        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                pred = student(images).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.numel()
        test_acc = correct / total if total else 0.0

        pseudo_acc_total = 0.0
        pseudo_acc_count = 0
        with torch.no_grad():
            for images, labels in unlabeled_eval:
                images, labels = images.to(device), labels.to(device)
                probs = torch.softmax(teacher(images), dim=1)
                conf, pred = torch.max(probs, dim=1)
                mask = conf >= tau
                if mask.any():
                    pseudo_acc_total += (pred[mask] == labels[mask]).float().sum().item()
                    pseudo_acc_count += mask.sum().item()
        pseudo_acc = pseudo_acc_total / pseudo_acc_count if pseudo_acc_count else 0.0
        accept = float(pseudo_acc_count) / float(len(unlabeled_eval.dataset))

        history.append(
            {
                "epoch": float(epoch),
                "test_acc": test_acc,
                "pseudo_label_acc": pseudo_acc,
                "accept_rate": accept,
            }
        )

    return HybridResult(history=history)
