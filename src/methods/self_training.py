"""Naive self-training loop (hard or soft labels)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from eval.eval_classification import evaluate_classification, evaluate_pseudo_labels
from utils.progress import progress
from utils.schedules import linear_rampup


@dataclass
class SelfTrainResult:
    history: List[Dict[str, float]]


def run_self_training(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    unlabeled_eval: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rounds: int,
    threshold: float,
    use_soft: bool,
    max_unlabeled_per_round: int,
    threshold_start: float | None = None,
    rampup_rounds: int = 0,
    use_progress: bool = False,
) -> SelfTrainResult:
    model.to(device)
    ce = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []

    for r in progress(range(rounds), enabled=use_progress, desc="self-train rounds"):
        if threshold_start is not None and rampup_rounds > 0:
            alpha = linear_rampup(r + 1, rampup_rounds)
            threshold_t = threshold_start + (threshold - threshold_start) * alpha
        else:
            threshold_t = threshold

        model.train()
        sup_loss_sum = 0.0
        sup_count = 0
        for images, labels in labeled_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sup_loss_sum += float(loss.item()) * labels.numel()
            sup_count += labels.numel()

        model.eval()
        pseudo_images = []
        pseudo_targets = []
        confidences = []
        with torch.no_grad():
            for images, labels in unlabeled_eval:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                mask = conf >= threshold_t if threshold_t > 0 else torch.ones_like(conf, dtype=torch.bool)
                if mask.any():
                    pseudo_images.append(images[mask].cpu())
                    pseudo_targets.append(pred[mask].cpu())
                    confidences.append(conf[mask].cpu())
                if sum(x.size(0) for x in pseudo_images) >= max_unlabeled_per_round:
                    break

        if pseudo_images:
            Xp = torch.cat(pseudo_images)
            yp = torch.cat(pseudo_targets)
            conf = torch.cat(confidences)
        else:
            Xp = torch.empty(0)
            yp = torch.empty(0, dtype=torch.long)
            conf = torch.empty(0)

        unsup_loss_total = 0.0
        unsup_steps = 0
        if Xp.numel() > 0:
            model.train()
            for _ in range(1):
                idx = torch.randperm(Xp.size(0))
                Xp_shuf = Xp[idx].to(device)
                yp_shuf = yp[idx].to(device)
                logits = model(Xp_shuf)
                if use_soft:
                    probs = torch.softmax(logits, dim=1)
                    y_onehot = torch.zeros_like(probs)
                    y_onehot.scatter_(1, yp_shuf.unsqueeze(1), 1.0)
                    loss = torch.mean(torch.sum(-y_onehot * torch.log(probs + 1e-8), dim=1))
                else:
                    loss = ce(logits, yp_shuf)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                unsup_loss_total += float(loss.item())
                unsup_steps += 1

        val = evaluate_classification(model, test_loader, device)
        pseudo_eval = evaluate_pseudo_labels(model, unlabeled_eval, device, threshold_t)
        train_sup_loss = sup_loss_sum / float(sup_count) if sup_count else 0.0
        train_unsup_loss = unsup_loss_total / float(unsup_steps) if unsup_steps else 0.0
        train_loss = train_sup_loss + train_unsup_loss
        state_error_e_t = (
            1.0 - pseudo_eval.pseudo_label_accuracy
            if pseudo_eval.pseudo_label_fraction > 0
            else float("nan")
        )

        history.append(
            {
                "round": float(r),
                "threshold": float(threshold_t),
                "train_loss": train_loss,
                "supervised_loss": train_sup_loss,
                "unsupervised_loss": train_unsup_loss,
                "val_loss": val.loss,
                "val_accuracy": val.accuracy,
                "test_acc": val.accuracy,
                "pseudo_label_fraction": pseudo_eval.pseudo_label_fraction,
                "pseudo_label_accuracy": pseudo_eval.pseudo_label_accuracy,
                "mean_confidence": pseudo_eval.mean_confidence,
                "entropy": pseudo_eval.entropy,
                # Backward-compatible aliases used by existing notebooks.
                "pseudo_label_acc": pseudo_eval.pseudo_label_accuracy,
                "accept_rate": pseudo_eval.pseudo_label_fraction,
                "state_error_e_t": state_error_e_t,
                "avg_conf_selected": float(conf.mean().item()) if conf.numel() > 0 else 0.0,
            }
        )

    return SelfTrainResult(history=history)
