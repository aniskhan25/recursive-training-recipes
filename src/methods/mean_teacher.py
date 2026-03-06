"""Mean Teacher SSL training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from eval.eval_classification import evaluate_classification
from utils.progress import progress
from utils.schedules import linear_rampup


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


def _ema_gap(student: nn.Module, teacher: nn.Module) -> float:
    total_sq = 0.0
    total_n = 0
    with torch.no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            diff = (s_param - t_param).detach()
            total_sq += float(torch.sum(diff * diff).item())
            total_n += diff.numel()
    if total_n == 0:
        return 0.0
    return float((total_sq / float(total_n)) ** 0.5)


def _teacher_pseudo_stats(
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    teacher.eval()
    accepted = 0
    total = 0
    correct = 0.0
    conf_sum = 0.0
    ent_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            u_w, _ = _split_unlabeled_views(images)
            u_w, labels = u_w.to(device), labels.to(device)
            probs = torch.softmax(teacher(u_w), dim=1)
            conf, pred = torch.max(probs, dim=1)
            mask = conf >= threshold if threshold > 0 else torch.ones_like(conf, dtype=torch.bool)
            if mask.any():
                accepted += int(mask.sum().item())
                correct += float((pred[mask] == labels[mask]).float().sum().item())
            conf_sum += float(conf.sum().item())
            ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_sum += float(ent.sum().item())
            total += labels.numel()
    if total == 0:
        return {
            "pseudo_label_fraction": 0.0,
            "pseudo_label_accuracy": 0.0,
            "mean_confidence": 0.0,
            "entropy": 0.0,
        }
    return {
        "pseudo_label_fraction": float(accepted) / float(total),
        "pseudo_label_accuracy": (correct / float(accepted)) if accepted else 0.0,
        "mean_confidence": conf_sum / float(total),
        "entropy": ent_sum / float(total),
    }


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
    unlabeled_eval: DataLoader | None = None,
    pseudo_threshold: float = 0.95,
    warmup_epochs: int = 0,
    use_progress: bool = False,
) -> MeanTeacherResult:
    student.to(device)
    teacher.to(device)
    teacher.load_state_dict(student.state_dict())
    teacher.eval()
    ce = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []

    for epoch in progress(range(epochs), enabled=use_progress, desc="mean teacher epochs"):
        student.train()
        lambda_u_t = lambda_u * linear_rampup(epoch + 1, warmup_epochs) if warmup_epochs > 0 else lambda_u
        train_loss_total = 0.0
        sup_loss_total = 0.0
        unsup_loss_total = 0.0
        train_steps = 0
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

            loss = loss_sup + lambda_u_t * loss_unsup
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            sup_loss_total += float(loss_sup.item())
            unsup_loss_total += float(loss_unsup.item())
            train_steps += 1

            _update_ema(teacher, student, ema_decay)

        val = evaluate_classification(student, test_loader, device)
        eval_loader = unlabeled_eval if unlabeled_eval is not None else unlabeled_loader
        pseudo = _teacher_pseudo_stats(teacher, eval_loader, device, pseudo_threshold)
        train_loss = train_loss_total / float(train_steps) if train_steps else 0.0
        sup_loss = sup_loss_total / float(train_steps) if train_steps else 0.0
        unsup_loss = unsup_loss_total / float(train_steps) if train_steps else 0.0

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
        ema_gap = _ema_gap(student, teacher)

        history.append(
            {
                "epoch": float(epoch),
                "lambda_u": float(lambda_u_t),
                "train_loss": train_loss,
                "supervised_loss": sup_loss,
                "unsupervised_loss": unsup_loss,
                "val_loss": val.loss,
                "val_accuracy": val.accuracy,
                "test_acc": val.accuracy,
                "pseudo_label_fraction": pseudo["pseudo_label_fraction"],
                "pseudo_label_accuracy": pseudo["pseudo_label_accuracy"],
                "mean_confidence": pseudo["mean_confidence"],
                "entropy": pseudo["entropy"],
                "teacher_student_disagreement": disagreement,
                "ema_gap": ema_gap,
                # Backward-compatible aliases used by existing notebooks.
                "pseudo_label_acc": pseudo["pseudo_label_accuracy"],
                "accept_rate": pseudo["pseudo_label_fraction"],
            }
        )

    return MeanTeacherResult(history=history)
