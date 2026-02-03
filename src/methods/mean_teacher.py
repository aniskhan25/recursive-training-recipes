"""Mean Teacher SSL training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import acceptance_rate, accuracy, teacher_student_disagreement


@dataclass
class MeanTeacherResult:
    history: List[Dict[str, float]]


def _update_ema(teacher: nn.Module, student: nn.Module, ema_decay: float) -> None:
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data = ema_decay * t_param.data + (1.0 - ema_decay) * s_param.data


def _split_unlabeled_views(u_images: torch.Tensor | tuple | list) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(u_images, (tuple, list)) and len(u_images) == 2:
        return u_images[0], u_images[1]
    return u_images, u_images


def run_mean_teacher(
    student: nn.Module,
    teacher: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    ema_decay: float,
    lambda_u: float,
    warmup_epochs: int = 0,
) -> MeanTeacherResult:
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

            u_w, u_s = _split_unlabeled_views(u_images)
            l_images, l_labels = l_images.to(device), l_labels.to(device)
            u_w, u_s = u_w.to(device), u_s.to(device)

            logits_l = student(l_images)
            loss_sup = ce(logits_l, l_labels)

            with torch.no_grad():
                t_logits = teacher(u_w)
                t_probs = torch.softmax(t_logits, dim=1)

            s_logits = student(u_s)
            s_probs = torch.softmax(s_logits, dim=1)
            loss_unsup = torch.mean((s_probs - t_probs) ** 2)

            if warmup_epochs > 0:
                lambda_u_t = lambda_u * min(1.0, float(epoch + 1) / warmup_epochs)
            else:
                lambda_u_t = lambda_u

            loss = loss_sup + lambda_u_t * loss_unsup
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

        disagreement_total = 0.0
        disagreement_count = 0
        with torch.no_grad():
            for images, _ in unlabeled_loader:
                u_w, u_s = _split_unlabeled_views(images)
                u_w, u_s = u_w.to(device), u_s.to(device)
                t_probs = torch.softmax(teacher(u_w), dim=1)
                s_probs = torch.softmax(student(u_s), dim=1)
                disagreement_total += torch.mean(torch.abs(t_probs - s_probs)).item() * u_w.size(0)
                disagreement_count += u_w.size(0)
        disagreement = disagreement_total / disagreement_count if disagreement_count else 0.0

        history.append({"epoch": float(epoch), "test_acc": test_acc, "teacher_student_disagreement": disagreement})

    return MeanTeacherResult(history=history)
