"""Download MNIST with torchvision and create the three-split local pickle.

This conversion reproduces the expected *format*, not necessarily the sample
ordering / split used in the original archived experiment.
"""
from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST


def main() -> None:
    parser = argparse.ArgumentParser(description="Create data/dataset/mnist.pkl.gz from torchvision MNIST")
    parser.add_argument("--data-dir", default="data/dataset")
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "mnist.pkl.gz"
    if output.exists() and not args.overwrite:
        parser.error(f"{output} already exists; use --overwrite to replace it")

    train = MNIST(root=root, train=True, download=True)
    test = MNIST(root=root, train=False, download=True)
    valid_size = args.validation_size
    if not 0 < valid_size < len(train):
        parser.error("--validation-size must be between 1 and 59999")

    def to_arrays(dataset):
        images = dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        labels = dataset.targets.numpy().astype(np.int64)
        return images, labels

    x, y = to_arrays(train)
    x_test, y_test = to_arrays(test)
    boundary = len(y) - valid_size
    splits = ((x[:boundary], y[:boundary]), (x[boundary:], y[boundary:]), (x_test, y_test))
    with gzip.open(output, "wb") as f:
        pickle.dump(splits, f, protocol=4)
    print(f"Saved {output} with train={boundary}, validation={valid_size}, test={len(y_test)}")
    print("Note: resulting split order may differ from the archived experiment's mnist.pkl.gz.")


if __name__ == "__main__":
    main()
