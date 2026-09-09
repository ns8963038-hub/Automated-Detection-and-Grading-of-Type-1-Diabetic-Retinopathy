"""Training and evaluation loops, plus the metrics used to grade the model."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             precision_recall_fscore_support)
from tqdm import tqdm

from src import config as C


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Metrics appropriate for an ordinal, imbalanced 5-grade problem.

    Quadratic weighted kappa is the headline number: unlike accuracy it
    penalises a grade-0 -> grade-4 mistake far more heavily than grade-3 ->
    grade-4, which matches how a clinician would judge the error.
    """
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "qwk": 0.0 if np.isnan(qwk) else qwk,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def referable_dr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Collapse the 5 grades into the binary screening decision.

    Grades 0-1 are monitored; grades 2-4 ("referable DR") need an
    ophthalmologist. This is the decision a screening tool actually drives, so
    its sensitivity is the number a clinical reader cares about most.
    """
    t = (y_true >= 2).astype(int)
    p = (y_pred >= 2).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        t, p, average="binary", zero_division=0
    )
    tn = int(((t == 0) & (p == 0)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    return {
        "referable_sensitivity": rec,
        "referable_precision": prec,
        "referable_f1": f1,
        "referable_specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(C.NUM_CLASSES)), zero_division=0
    )
    return {
        C.CLASS_NAMES[i]: {
            "precision": float(prec[i]), "recall": float(rec[i]),
            "f1": float(f1[i]), "support": int(sup[i]),
        }
        for i in range(C.NUM_CLASSES)
    }


# --------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------
def cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int,
                       steps_per_epoch: int):
    """Linear warmup then cosine decay, stepped per batch.

    Warmup matters here because the classification head is randomly
    initialised while the backbone is pretrained -- a large LR in the first
    few hundred steps would wash out the ImageNet features.
    """
    warmup_steps = max(warmup_epochs * steps_per_epoch, 1)
    total_steps = max(total_epochs * steps_per_epoch, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------
# Loops
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device,
                    scheduler=None, epoch: int = 0, grad_clip: float = C.GRAD_CLIP):
    model.train()
    running_loss, seen = 0.0, 0
    preds_all, labels_all = [], []

    pbar = tqdm(loader, desc=f"epoch {epoch:02d} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        seen += bs
        preds_all.append(outputs.detach().argmax(1).cpu())
        labels_all.append(labels.detach().cpu())
        pbar.set_postfix(loss=f"{running_loss / seen:.4f}")

    y_pred = torch.cat(preds_all).numpy()
    y_true = torch.cat(labels_all).numpy()
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / max(seen, 1)
    return metrics


# --------------------------------------------------------------------------
# Test-time augmentation
# --------------------------------------------------------------------------
def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """One of the eight symmetries of a square: 4 rotations x optional flip.

    Safe for fundus photographs specifically because they have no canonical
    orientation - left and right eyes are mirror images of each other, and the
    camera rotation is arbitrary. The same set would be wrong for, say, chest
    x-rays, where left and right are clinically distinct.
    """
    if k >= 4:
        x = torch.flip(x, dims=[3])
    return torch.rot90(x, k % 4, dims=[2, 3])


@torch.no_grad()
def tta_probabilities(model, images: torch.Tensor, n_aug: int = 8) -> torch.Tensor:
    """Average softmax probabilities over `n_aug` dihedral views.

    Averaging probabilities rather than logits: logits are unnormalised, so one
    view that happens to be confident would dominate the mean rather than
    contributing one vote.
    """
    total = None
    for k in range(n_aug):
        probs = torch.softmax(model(dihedral(images, k)), dim=1)
        total = probs if total is None else total + probs
    return total / n_aug


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "val",
             tta: int = 0):
    model.eval()
    running_loss, seen = 0.0, 0
    logits_all, labels_all = [], []

    for images, labels in tqdm(loader, desc=f"           [{desc}]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        seen += bs
        # With TTA the stored scores are already averaged probabilities; the
        # loss above stays on the single un-augmented view so it remains
        # comparable to training.
        scores = (tta_probabilities(model, images, tta) if tta
                  else torch.softmax(outputs, dim=1))
        logits_all.append(scores.cpu())
        labels_all.append(labels.cpu())

    probs_t = torch.cat(logits_all)
    y_true = torch.cat(labels_all).numpy()
    y_pred = probs_t.argmax(1).numpy()
    probs = probs_t.numpy()

    metrics = compute_metrics(y_true, y_pred)
    metrics.update(referable_dr_metrics(y_true, y_pred))
    metrics["loss"] = running_loss / max(seen, 1)
    return metrics, y_true, y_pred, probs


# --------------------------------------------------------------------------
# Optimizers
#
# The optimizer decides how gradients become weight updates. This is a
# separate concern from the activation function, which is the non-linearity
# inside the network -- a run always uses one of each.
# --------------------------------------------------------------------------
def build_optimizer(name: str, params, lr: float, weight_decay: float):
    """Construct an optimizer by name.

    AdamW is the default: it adapts a learning rate per parameter, which
    converges reliably on a dataset this small, and its decoupled weight decay
    regularises correctly (plain Adam folds decay into the gradient, which
    interacts badly with the adaptive scaling). SGD with momentum is the
    classic alternative -- it often generalises slightly better but needs a
    far more carefully tuned schedule.
    """
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=C.SGD_MOMENTUM,
                               weight_decay=weight_decay, nesterov=True)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay,
                                   momentum=C.SGD_MOMENTUM)
    raise KeyError(f"Unknown optimizer '{name}'. "
                   f"Options: adamw, adam, sgd, rmsprop")


def build_criterion(device, weights: torch.Tensor | None = None):
    if weights is not None:
        weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=C.LABEL_SMOOTHING)
