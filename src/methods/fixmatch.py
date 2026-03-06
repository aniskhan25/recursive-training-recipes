"""FixMatch-style SSL training step."""

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
class FixMatchResult:
    history: List[Dict[str, float]]


def _split_unlabeled_views(u_images: torch.Tensor | tuple | list) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(u_images, (tuple, list)) and len(u_images) == 2:
        return u_images[0], u_images[1]
    return u_images, u_images


def run_fixmatch(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    unlabeled_eval: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    tau: float,
    lambda_u: float,
    tau_start: float | None = None,
    rampup_epochs: int = 0,
    use_progress: bool = False,
) -> FixMatchResult:
    model.to(device)
    ce = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []

    for epoch in progress(range(epochs), enabled=use_progress, desc="fixmatch epochs"):
        if tau_start is not None and rampup_epochs > 0:
            tau_t = tau_start + (tau - tau_start) * linear_rampup(epoch + 1, rampup_epochs)
        else:
            tau_t = tau
        lambda_u_t = lambda_u * linear_rampup(epoch + 1, rampup_epochs) if rampup_epochs > 0 else lambda_u

        model.train()
        train_loss_total = 0.0
        train_steps = 0
        sup_loss_total = 0.0
        unsup_loss_total = 0.0
        accepted_batches = 0
        accepted_total = 0
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

            logits_l = model(l_images)
            loss_sup = ce(logits_l, l_labels)

            with torch.no_grad():
                logits_u_w = model(u_w)
                probs_u = torch.softmax(logits_u_w, dim=1)
                conf, pseudo = torch.max(probs_u, dim=1)
                mask = conf >= tau_t

            logits_u_s = model(u_s)
            loss_unsup = ce(logits_u_s[mask], pseudo[mask]) if mask.any() else torch.tensor(0.0, device=device)

            loss = loss_sup + lambda_u_t * loss_unsup
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            sup_loss_total += float(loss_sup.item())
            unsup_loss_total += float(loss_unsup.item())
            train_steps += 1
            accepted_total += int(mask.sum().item())
            accepted_batches += mask.numel()

        val = evaluate_classification(model, test_loader, device)
        pseudo_eval = evaluate_pseudo_labels(model, unlabeled_eval, device, tau_t)
        train_loss = train_loss_total / float(train_steps) if train_steps else 0.0
        sup_loss = sup_loss_total / float(train_steps) if train_steps else 0.0
        unsup_loss = unsup_loss_total / float(train_steps) if train_steps else 0.0
        batch_accept = float(accepted_total) / float(accepted_batches) if accepted_batches else 0.0

        history.append(
            {
                "epoch": float(epoch),
                "threshold": float(tau_t),
                "lambda_u": float(lambda_u_t),
                "train_loss": train_loss,
                "supervised_loss": sup_loss,
                "unsupervised_loss": unsup_loss,
                "val_loss": val.loss,
                "val_accuracy": val.accuracy,
                "test_acc": val.accuracy,
                "pseudo_label_fraction": pseudo_eval.pseudo_label_fraction,
                "pseudo_label_accuracy": pseudo_eval.pseudo_label_accuracy,
                "mean_confidence": pseudo_eval.mean_confidence,
                "entropy": pseudo_eval.entropy,
                "batch_accept_rate": batch_accept,
                # Backward-compatible aliases used by existing notebooks.
                "pseudo_label_acc": pseudo_eval.pseudo_label_accuracy,
                "accept_rate": pseudo_eval.pseudo_label_fraction,
            }
        )

    return FixMatchResult(history=history)
