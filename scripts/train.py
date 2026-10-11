"""Training utilities and CLI helpers for the CNN architecture study."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.data_loader import make_mnist_dataloaders
from models.model import build_model
from utils.metrics import accuracy_from_logits, count_trainable_parameters


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Make experiments more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Pick CUDA, Apple MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> dict[str, float]:
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        batch_size = labels.size(0)
        correct, total = accuracy_from_logits(logits, labels)
        running_loss += loss.item() * batch_size
        running_correct += correct
        running_total += total

    return {"loss": running_loss / running_total, "acc": running_correct / running_total}


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model on a validation or test loader."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        correct, total = accuracy_from_logits(logits, labels)
        running_loss += loss.item() * batch_size
        running_correct += correct
        running_total += total

    return {"loss": running_loss / running_total, "acc": running_correct / running_total}


def save_checkpoint(
    path: str | os.PathLike,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_valid_loss: float,
    history: list[dict],
    extra: dict | None = None,
) -> None:
    """Save model, optimizer, scheduler, and history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": model.__class__.__name__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_valid_loss": best_valid_loss,
            "history": history,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str | os.PathLike,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    device: torch.device | None = None,
):
    """Load a checkpoint into ``model`` and optionally optimizer/scheduler."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    checkpoint_path: str | os.PathLike = "models/saved_models/best_model.pt",
    grad_clip: float | None = None,
    scheduler_patience: int = 2,
    early_stopping_patience: int | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Train a model and save the best checkpoint by validation loss."""
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=scheduler_patience,
    )

    history: list[dict] = []
    best_valid_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    if verbose:
        print(f"Device: {device}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Trainable parameters: {count_trainable_parameters(model):,}")
        print("-" * 72)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
        )
        valid_metrics = evaluate_epoch(model, valid_loader, criterion, device)
        scheduler.step(valid_metrics["loss"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "valid_loss": float(valid_metrics["loss"]),
            "valid_acc": float(valid_metrics["acc"]),
        }
        history.append(row)

        improved = valid_metrics["loss"] < best_valid_loss
        if improved:
            best_valid_loss = float(valid_metrics["loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_valid_loss=best_valid_loss,
                history=history,
                extra={"lr": lr, "weight_decay": weight_decay, "grad_clip": grad_clip},
            )
        else:
            epochs_without_improvement += 1

        if verbose:
            marker = "  <-- best" if improved else ""
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"lr {row['lr']:.2e} | "
                f"train loss {row['train_loss']:.4f}, acc {row['train_acc']:.4f} | "
                f"valid loss {row['valid_loss']:.4f}, acc {row['valid_acc']:.4f}"
                f"{marker}"
            )

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            if verbose:
                print(f"Early stopping after {epoch} epochs. Best epoch was {best_epoch}.")
            break

    history_df = pd.DataFrame(history)
    return history_df, {
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "checkpoint_path": str(checkpoint_path),
    }


def train_and_compare_models(
    model_factories: dict[str, Callable[[], nn.Module]],
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    checkpoint_dir: str | os.PathLike = "models/saved_models",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Train several models with identical hyperparameters and compare them."""
    if device is None:
        device = get_device()

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    histories = {}
    rows = []
    for model_name, make_model in model_factories.items():
        print("\n" + "=" * 80)
        print(f"Training {model_name}")
        print("=" * 80)

        set_seed(42)
        model = make_model()
        ckpt_path = checkpoint_dir / f"{model_name.lower().replace(' ', '_')}_best.pt"
        history_df, best_info = fit_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            checkpoint_path=ckpt_path,
            scheduler_patience=2,
            verbose=True,
        )

        test_model = make_model().to(device)
        checkpoint = load_checkpoint(ckpt_path, model=test_model, device=device)
        test_metrics = evaluate_epoch(test_model, test_loader, criterion, device)
        histories[model_name] = history_df
        rows.append(
            {
                "model": model_name,
                "parameters": count_trainable_parameters(make_model()),
                "best_epoch": checkpoint["epoch"],
                "best_valid_loss": best_info["best_valid_loss"],
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["acc"],
                "checkpoint": str(ckpt_path),
            }
        )

    results_df = pd.DataFrame(rows).sort_values("test_acc", ascending=False).reset_index(drop=True)
    return results_df, histories


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one MNIST model from the command line.")
    parser.add_argument("--model", default="residual", choices=["residual", "inception", "resnext", "custom"])
    parser.add_argument("--data-path", default="data/dataset/mnist.pkl.gz")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--seed", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    device = get_device()
    checkpoint = args.checkpoint or f"models/saved_models/{args.model}_best.pt"
    train_loader, valid_loader, _test_loader = make_mnist_dataloaders(
        path=args.data_path,
        batch_size=args.batch_size,
        flatten=False,
        pin_memory=(device.type == "cuda"),
    )
    model = build_model(args.model, num_classes=10, dropout=args.dropout, groups=args.groups)
    history_df, best_info = fit_model(
        model,
        train_loader,
        valid_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        checkpoint_path=checkpoint,
    )
    history_path = Path(checkpoint).with_suffix(".history.csv")
    history_df.to_csv(history_path, index=False)
    print(best_info)
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
