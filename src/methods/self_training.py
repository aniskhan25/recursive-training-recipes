"""Naive self-training loop (hard or soft labels)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import acceptance_rate, accuracy, entropy_from_probs, pseudo_label_error


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
) -> SelfTrainResult:
    model.to(device)
    ce = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []

    for r in range(rounds):
        model.train()
        for images, labels in labeled_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = ce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        pseudo_images = []
        pseudo_targets = []
        pseudo_truth = []
        confidences = []
        with torch.no_grad():
            for images, labels in unlabeled_eval:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                mask = conf >= threshold if threshold > 0 else torch.ones_like(conf, dtype=torch.bool)
                if mask.any():
                    pseudo_images.append(images[mask].cpu())
                    pseudo_targets.append(pred[mask].cpu())
                    pseudo_truth.append(labels[mask].cpu())
                    confidences.append(conf[mask].cpu())
                if sum(x.size(0) for x in pseudo_images) >= max_unlabeled_per_round:
                    break

        if pseudo_images:
            Xp = torch.cat(pseudo_images)
            yp = torch.cat(pseudo_targets)
            yt = torch.cat(pseudo_truth)
            conf = torch.cat(confidences)
            e_t = pseudo_label_error(yt, yp)
            acc_pseudo = accuracy(yt, yp)
            accept = float(Xp.size(0)) / float(len(unlabeled_eval.dataset))
        else:
            Xp = torch.empty(0)
            yp = torch.empty(0, dtype=torch.long)
            yt = torch.empty(0, dtype=torch.long)
            conf = torch.empty(0)
            e_t = 0.0
            acc_pseudo = 0.0
            accept = 0.0

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

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.numel()
        test_acc = correct / total if total else 0.0

        history.append(
            {
                "round": float(r),
                "test_acc": test_acc,
                "pseudo_label_acc": acc_pseudo,
                "accept_rate": accept,
                "state_error_e_t": e_t,
                "avg_conf_selected": float(conf.mean().item()) if conf.numel() > 0 else 0.0,
            }
        )

    return SelfTrainResult(history=history)
