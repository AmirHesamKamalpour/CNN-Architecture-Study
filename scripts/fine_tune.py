"""Head-only Fashion-MNIST fine-tuning from a trained MNIST checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn as nn

from data.data_loader import FASHION_MNIST_CLASSES, make_fashion_mnist_dataloaders
from models.model import build_model
from scripts.evaluate import collect_predictions, evaluate_best_checkpoint
from scripts.train import evaluate_epoch, get_device, load_checkpoint, set_seed
from utils.metrics import count_parameters, most_confused_pairs_table, per_class_accuracy_table
from utils.visualization import plot_accuracy_curve, plot_confusion_matrix, plot_loss_curve


def reset_classification_head(model: nn.Module) -> None:
    """Reset trainable layers inside ``model.head``."""
    for module in model.head.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


def freeze_all_except_head(model: nn.Module) -> None:
    """Freeze the feature extractor and keep the classification head trainable."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True


def set_backbone_eval_head_train(model: nn.Module) -> None:
    """Keep BatchNorm in the frozen backbone in eval mode while training the head."""
    model.eval()
    if hasattr(model, "stem"):
        model.stem.eval()
    if hasattr(model, "blocks"):
        model.blocks.eval()
    if hasattr(model, "head"):
        model.head.train()


def train_one_epoch_head_only(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train only the classifier head for one epoch."""
    set_backbone_eval_head_train(model)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return {"loss": total_loss / total_examples, "acc": total_correct / total_examples}


def fit_head_only(
    model: nn.Module,
    train_loader,
    valid_loader,
    *,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_path: str | Path = "models/saved_models/best_residual_fashionmnist_head_only.pt",
) -> pd.DataFrame:
    """Fine-tune the classifier head and save the best validation checkpoint."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    history = []
    best_valid_loss = float("inf")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_head_only(model, train_loader, criterion, optimizer, device)
        valid_metrics = evaluate_epoch(model, valid_loader, criterion, device)
        scheduler.step(valid_metrics["loss"])

        row = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "valid_loss": float(valid_metrics["loss"]),
            "valid_acc": float(valid_metrics["acc"]),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        marker = ""
        if row["valid_loss"] < best_valid_loss:
            best_valid_loss = row["valid_loss"]
            marker = "  <-- best"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_valid_loss": best_valid_loss,
                    "history": history,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"lr {row['lr']:.2e} | "
            f"train loss {row['train_loss']:.4f}, acc {row['train_acc']:.4f} | "
            f"valid loss {row['valid_loss']:.4f}, acc {row['valid_acc']:.4f}"
            f"{marker}"
        )

    return pd.DataFrame(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an MNIST checkpoint on Fashion-MNIST.")
    parser.add_argument("--pretrained-checkpoint", default="models/saved_models/residual_best.pt")
    parser.add_argument("--output-checkpoint", default="models/saved_models/fashion_head_only_best.pt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, valid_loader, test_loader = make_fashion_mnist_dataloaders(
        root="data/dataset",
        batch_size=args.batch_size,
        valid_size=5000,
        seed=args.seed,
        normalize=True,
    )

    model = build_model("residual", num_classes=10, dropout=0.05).to(device)
    load_checkpoint(args.pretrained_checkpoint, model=model, device=device)
    reset_classification_head(model)
    freeze_all_except_head(model)
    print(count_parameters(model))

    history_df = fit_head_only(
        model,
        train_loader,
        valid_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=args.output_checkpoint,
    )
    history_path = Path(args.output_checkpoint).with_suffix(".history.csv")
    history_df.to_csv(history_path, index=False)

    test_metrics, checkpoint = evaluate_best_checkpoint(model, args.output_checkpoint, test_loader, device)
    print("=" * 70)
    print("Fashion-MNIST Head-Only Fine-Tuning Test Results")
    print(f"Best checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test loss:            {test_metrics['loss']:.4f}")
    print(f"Test accuracy:        {test_metrics['acc']:.4f}")

    y_true, y_pred = collect_predictions(model, test_loader, device=device)
    print("\nWorst classes by per-class accuracy:")
    print(per_class_accuracy_table(y_true, y_pred, FASHION_MNIST_CLASSES).head(5))
    print("\nMost confused class pairs:")
    print(most_confused_pairs_table(y_true, y_pred, FASHION_MNIST_CLASSES, top_k=10))

    if args.plots:
        plot_loss_curve(history_df, title="Fashion-MNIST head-only fine-tuning loss")
        plot_accuracy_curve(history_df, title="Fashion-MNIST head-only fine-tuning accuracy")
        plot_confusion_matrix(y_true, y_pred, class_names=FASHION_MNIST_CLASSES, title="Fashion-MNIST confusion matrix")


if __name__ == "__main__":
    main()
