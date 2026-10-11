"""Evaluation helpers and an evaluation CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.data_loader import make_mnist_dataloaders
from models.model import build_model
from scripts.train import evaluate_epoch, get_device, load_checkpoint


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> tuple:
    """Collect true and predicted labels for an entire loader."""
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


def evaluate_best_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    test_loader: DataLoader,
    device: torch.device | None = None,
) -> tuple[dict[str, float], dict]:
    """Load a checkpoint into ``model`` and evaluate it on ``test_loader``."""
    if device is None:
        device = get_device()

    model = model.to(device)
    checkpoint = load_checkpoint(checkpoint_path, model=model, device=device)
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_epoch(model, test_loader, criterion, device)
    return test_metrics, checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved MNIST checkpoint.")
    parser.add_argument("--model", default="residual", choices=["residual", "inception", "resnext", "custom"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="data/dataset/mnist.pkl.gz")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--groups", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = get_device()
    _train_loader, _valid_loader, test_loader = make_mnist_dataloaders(
        path=args.data_path,
        batch_size=args.batch_size,
        flatten=False,
        pin_memory=(device.type == "cuda"),
    )
    model = build_model(args.model, num_classes=10, dropout=args.dropout, groups=args.groups)
    test_metrics, checkpoint = evaluate_best_checkpoint(model, args.checkpoint, test_loader, device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Test loss: {test_metrics['loss']:.4f}")
    print(f"Test accuracy: {test_metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
