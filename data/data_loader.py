"""Data loading utilities for MNIST and Fashion-MNIST."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """Custom dataset for the classic ``mnist.pkl.gz`` format.

    The pickle file is expected to contain three splits:
    ``data[0]`` = train, ``data[1]`` = validation, and ``data[2]`` = test.
    """

    SPLIT_TO_INDEX = {
        "train": 0,
        "valid": 1,
        "validation": 1,
        "val": 1,
        "test": 2,
    }

    def __init__(
        self,
        path: str | Path,
        split: str = "train",
        flatten: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        if split not in self.SPLIT_TO_INDEX:
            raise ValueError(
                f"Unknown split '{split}'. Expected one of {list(self.SPLIT_TO_INDEX)}."
            )

        self.path = Path(path)
        self.split = split
        self.flatten = flatten
        self.transform = transform
        self.target_transform = target_transform

        if not self.path.exists():
            raise FileNotFoundError(
                f"Could not find MNIST file at {self.path}. "
                "Put mnist.pkl.gz in data/dataset/ or update config/config.yaml."
            )

        with gzip.open(self.path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        images, labels = data[self.SPLIT_TO_INDEX[split]]
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()

        if not flatten:
            self.images = self.images.view(-1, 1, 28, 28)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


def make_mnist_dataloaders(
    path: str | Path,
    batch_size: int = 64,
    flatten: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for ``mnist.pkl.gz``."""

    train_dataset = MNISTDataset(path=path, split="train", flatten=flatten)
    valid_dataset = MNISTDataset(path=path, split="valid", flatten=flatten)
    test_dataset = MNISTDataset(path=path, split="test", flatten=flatten)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader, test_loader


FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530


def make_fashion_mnist_dataloaders(
    root: str | Path = "data/dataset",
    batch_size: int = 256,
    valid_size: int = 5000,
    num_workers: int = 0,
    seed: int = 42,
    normalize: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for Fashion-MNIST."""

    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=(FASHION_MNIST_MEAN,), std=(FASHION_MNIST_STD,))
        )
    transform = transforms.Compose(transform_list)

    root = Path(root)
    full_train_dataset = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    if not 0 < valid_size < len(full_train_dataset):
        raise ValueError(
            f"valid_size must be between 1 and {len(full_train_dataset) - 1}; got {valid_size}."
        )

    train_size = len(full_train_dataset) - valid_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, valid_dataset = random_split(
        full_train_dataset,
        lengths=[train_size, valid_size],
        generator=generator,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader, test_loader
