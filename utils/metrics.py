"""Metrics and summary tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that require gradients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total, trainable, and frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = count_trainable_parameters(model)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """Return ``(number_correct, batch_size)`` from class logits and labels."""
    preds = logits.argmax(dim=1)
    return int((preds == labels).sum().item()), int(labels.size(0))


def confusion_matrix_numpy(y_true, y_pred, num_classes: int = 10) -> np.ndarray:
    """Small dependency-light confusion matrix implementation."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm


def per_class_accuracy_table(y_true, y_pred, class_names: list[str]) -> pd.DataFrame:
    """Build a table of per-class accuracy sorted from worst to best."""
    cm = confusion_matrix_numpy(y_true, y_pred, num_classes=len(class_names))
    correct = np.diag(cm)
    total = cm.sum(axis=1)
    acc = np.divide(
        correct,
        total,
        out=np.zeros_like(correct, dtype=float),
        where=total != 0,
    )
    return pd.DataFrame(
        {
            "class_id": np.arange(len(class_names)),
            "class_name": class_names,
            "correct": correct,
            "total": total,
            "accuracy": acc,
        }
    ).sort_values("accuracy", ascending=True).reset_index(drop=True)


def most_confused_pairs_table(y_true, y_pred, class_names: list[str], top_k: int = 10) -> pd.DataFrame:
    """Return the most frequent off-diagonal confusion pairs."""
    cm = confusion_matrix_numpy(y_true, y_pred, num_classes=len(class_names))
    rows = []
    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count > 0:
                rows.append(
                    {
                        "true_class": class_names[true_idx],
                        "predicted_class": class_names[pred_idx],
                        "count": count,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["true_class", "predicted_class", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(top_k).reset_index(drop=True)
