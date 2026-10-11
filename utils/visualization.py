"""Visualization helpers for training curves, confusion matrices, and activations."""

from __future__ import annotations

import math
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from utils.metrics import confusion_matrix_numpy


def plot_loss_curve(history_df: pd.DataFrame, title: str = "Loss curve") -> None:
    """Plot training and validation loss."""
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train loss")
    plt.plot(history_df["epoch"], history_df["valid_loss"], label="valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_accuracy_curve(history_df: pd.DataFrame, title: str = "Accuracy curve") -> None:
    """Plot training and validation accuracy."""
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_acc"], label="train acc")
    plt.plot(history_df["epoch"], history_df["valid_acc"], label="valid acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list[str] | None = None,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> None:
    """Plot a confusion matrix."""
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    cm = confusion_matrix_numpy(y_true, y_pred, num_classes=len(class_names))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_to_plot = cm / row_sums.clip(min=1)
        fmt = ".2f"
    else:
        cm_to_plot = cm
        fmt = "d"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_to_plot,
        annot=True,
        fmt=fmt,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def find_first_conv2d(model: nn.Module) -> nn.Conv2d:
    """Return the first convolution layer in a model."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No nn.Conv2d layer found in model.")


def get_random_batch_image(loader, device: torch.device) -> tuple[torch.Tensor, int]:
    """Grab one random image from the first batch of a loader."""
    images, labels = next(iter(loader))
    idx = random.randint(0, images.size(0) - 1)
    return images[idx : idx + 1].to(device), int(labels[idx].item())


def get_layer_activation(model: nn.Module, layer: nn.Module, x: torch.Tensor):
    """Run a forward pass and capture one layer activation."""
    activation = {}

    def hook_fn(_module, _inputs, output):
        activation["value"] = output.detach().cpu()

    handle = layer.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    handle.remove()

    pred = int(logits.argmax(dim=1).item())
    probs = torch.softmax(logits, dim=1).detach().cpu().squeeze(0)
    return activation["value"], pred, probs


def plot_first_conv_activations(
    model: nn.Module,
    test_loader,
    device: torch.device,
    model_name: str = "model",
    max_channels: int = 32,
    ncols: int = 8,
) -> None:
    """Visualize first convolution feature maps for one random test image."""
    first_conv = find_first_conv2d(model)
    image, true_label = get_random_batch_image(test_loader, device)
    activations, pred_label, probs = get_layer_activation(model, first_conv, image)
    act = activations[0]
    num_channels = min(act.size(0), max_channels)

    print(f"Model: {model_name}")
    print(f"First conv layer: {first_conv}")
    print(f"Input image shape: {tuple(image.shape)}")
    print(f"Activation shape: {tuple(activations.shape)}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {pred_label}")
    print(f"Prediction confidence: {probs[pred_label].item():.4f}")

    plt.figure(figsize=(3, 3))
    plt.imshow(image.detach().cpu().squeeze(), cmap="gray")
    plt.title(f"{model_name}\nTrue: {true_label}, Pred: {pred_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    nrows = math.ceil(num_channels / ncols)
    plt.figure(figsize=(ncols * 2, nrows * 2))
    for ch in range(num_channels):
        fmap = act[ch]
        fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        plt.subplot(nrows, ncols, ch + 1)
        plt.imshow(fmap_norm, cmap="viridis")
        plt.title(f"Filter {ch}")
        plt.axis("off")

    plt.suptitle(f"{model_name}: first conv activation maps", fontsize=16)
    plt.tight_layout()
    plt.show()
