"""Main entry point for the CNN architecture-comparison pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data.data_loader import make_mnist_dataloaders
from models.model import build_model
from scripts.evaluate import evaluate_best_checkpoint
from scripts.train import fit_model, get_device, set_seed, train_and_compare_models
from utils.visualization import plot_accuracy_curve, plot_loss_curve

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install PyYAML first: pip install pyyaml") from exc


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single_model(config: dict, model_name: str, make_plots: bool = False) -> None:
    """Train and evaluate one model."""
    seed = int(config["training"].get("seed", 9))
    set_seed(seed)
    device = get_device()

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    output_cfg = config["outputs"]

    train_loader, valid_loader, test_loader = make_mnist_dataloaders(
        path=data_cfg["mnist_path"],
        batch_size=train_cfg["batch_size"],
        flatten=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    dropout = model_cfg.get("custom_dropout", model_cfg.get("dropout", 0.05)) if model_name == "custom" else model_cfg.get("dropout", 0.05)
    model = build_model(
        model_name,
        num_classes=model_cfg.get("num_classes", 10),
        dropout=dropout,
        groups=model_cfg.get("groups", 4),
    )

    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "models/saved_models"))
    checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
    history_df, best_info = fit_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=train_cfg["epochs"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        device=device,
        checkpoint_path=checkpoint_path,
        grad_clip=train_cfg.get("grad_clip"),
        scheduler_patience=train_cfg.get("scheduler_patience", 2),
        early_stopping_patience=train_cfg.get("early_stopping_patience"),
        verbose=True,
    )

    history_path = checkpoint_path.with_suffix(".history.csv")
    history_df.to_csv(history_path, index=False)

    test_model = build_model(
        model_name,
        num_classes=model_cfg.get("num_classes", 10),
        dropout=dropout,
        groups=model_cfg.get("groups", 4),
    )
    test_metrics, checkpoint = evaluate_best_checkpoint(
        model=test_model,
        checkpoint_path=best_info["checkpoint_path"],
        test_loader=test_loader,
        device=device,
    )

    print("=" * 72)
    print(f"Best checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation loss:       {checkpoint['best_valid_loss']:.4f}")
    print(f"Test loss:             {test_metrics['loss']:.4f}")
    print(f"Test accuracy:         {test_metrics['acc']:.4f}")
    print(f"Saved checkpoint:      {best_info['checkpoint_path']}")
    print(f"Saved history:         {history_path}")

    if make_plots:
        plot_loss_curve(history_df, title=f"{model_name} loss")
        plot_accuracy_curve(history_df, title=f"{model_name} accuracy")


def run_comparison(config: dict) -> None:
    """Train and compare all model families."""
    set_seed(int(config["training"].get("seed", 9)))
    device = get_device()
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    output_cfg = config["outputs"]

    train_loader, valid_loader, test_loader = make_mnist_dataloaders(
        path=data_cfg["mnist_path"],
        batch_size=train_cfg["batch_size"],
        flatten=False,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
    )

    factories = {
        "residual": lambda: build_model("residual", dropout=model_cfg.get("dropout", 0.05)),
        "inception": lambda: build_model("inception", dropout=model_cfg.get("dropout", 0.05)),
        "resnext": lambda: build_model(
            "resnext",
            dropout=model_cfg.get("dropout", 0.05),
            groups=model_cfg.get("groups", 4),
        ),
        "custom": lambda: build_model(
            "custom",
            dropout=model_cfg.get("custom_dropout", 0.2),
        ),
    }
    results_df, histories = train_and_compare_models(
        model_factories=factories,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        epochs=train_cfg["epochs"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        device=device,
        checkpoint_dir=output_cfg.get("checkpoint_dir", "models/saved_models"),
    )

    results_path = Path(output_cfg.get("checkpoint_dir", "models/saved_models")) / "comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(results_df)
    print(f"Saved comparison results to {results_path}")

    for name, history_df in histories.items():
        history_path = Path(output_cfg.get("checkpoint_dir", "models/saved_models")) / f"{name}_history.csv"
        history_df.to_csv(history_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MNIST architecture-comparison pipeline.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", default=None, choices=["residual", "inception", "resnext", "custom"])
    parser.add_argument("--compare", action="store_true", help="Train and compare all models.")
    parser.add_argument("--plots", action="store_true", help="Show loss and accuracy plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.compare:
        run_comparison(config)
    else:
        model_name = args.model or config["model"].get("name", "residual")
        run_single_model(config, model_name=model_name, make_plots=args.plots)


if __name__ == "__main__":
    main()
