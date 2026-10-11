"""CNN architectures for the MNIST deep learning architecture-comparison project."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm2d -> optional ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_ch),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Basic residual block with an optional projection skip path."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ConvBNAct(in_ch, out_ch, kernel_size=3, stride=stride),
            ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, activation=False),
        )
        self.skip = (
            ConvBNAct(
                in_ch,
                out_ch,
                kernel_size=1,
                stride=stride,
                padding=0,
                activation=False,
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class ResNeXtBlock(nn.Module):
    """ResNeXt-style bottleneck block with grouped convolution."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bottleneck_ch: int,
        groups: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if bottleneck_ch % groups != 0:
            raise ValueError(
                f"bottleneck_ch must be divisible by groups. "
                f"Got bottleneck_ch={bottleneck_ch}, groups={groups}."
            )

        self.main = nn.Sequential(
            ConvBNAct(in_ch, bottleneck_ch, kernel_size=1, stride=1, padding=0),
            ConvBNAct(
                bottleneck_ch,
                bottleneck_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            ),
            ConvBNAct(
                bottleneck_ch,
                out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=False,
            ),
        )
        self.skip = (
            ConvBNAct(
                in_ch,
                out_ch,
                kernel_size=1,
                stride=stride,
                padding=0,
                activation=False,
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)
        self.out_channels = out_ch
        self.bottleneck_ch = bottleneck_ch
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.skip(x))


class InceptionFactorizedBlock(nn.Module):
    """Inception-style parallel-branch filter-concat block."""

    def __init__(
        self,
        in_ch: int,
        pool_proj: int = 96,
        out_1x1: int = 96,
        reduce_3x3: int = 64,
        out_3x3: int = 96,
        reduce_double_3x3: int = 64,
        out_double_3x3: int = 96,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=stride, padding=1, count_include_pad=False),
            ConvBNAct(in_ch, pool_proj, kernel_size=1, stride=1, padding=0),
        )
        self.branch_1x1 = ConvBNAct(in_ch, out_1x1, kernel_size=1, stride=stride, padding=0)
        self.branch_3x3 = nn.Sequential(
            ConvBNAct(in_ch, reduce_3x3, kernel_size=1, stride=1, padding=0),
            ConvBNAct(reduce_3x3, out_3x3, kernel_size=3, stride=stride, padding=1),
        )
        self.branch_double_3x3 = nn.Sequential(
            ConvBNAct(in_ch, reduce_double_3x3, kernel_size=1, stride=1, padding=0),
            ConvBNAct(
                reduce_double_3x3,
                out_double_3x3,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            ConvBNAct(out_double_3x3, out_double_3x3, kernel_size=3, stride=1, padding=1),
        )
        self.out_channels = pool_proj + out_1x1 + out_3x3 + out_double_3x3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.branch_pool(x),
                self.branch_1x1(x),
                self.branch_3x3(x),
                self.branch_double_3x3(x),
            ],
            dim=1,
        )


class LinearHead(nn.Module):
    """Global-average-pooling classification head."""

    def __init__(self, in_ch: int, num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1)]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(in_ch, num_classes))
        self.head = nn.Sequential(*layers)
        self.out_channels = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def make_resnext_block(
    in_ch: int,
    out_ch: int,
    bottleneck_ch: int | None = None,
    groups: int = 4,
    stride: int = 1,
) -> ResNeXtBlock:
    """Create a ResNeXt block with a safe default bottleneck width."""

    bottleneck_ch = bottleneck_ch if bottleneck_ch is not None else out_ch // 2
    return ResNeXtBlock(
        in_ch=in_ch,
        out_ch=out_ch,
        bottleneck_ch=bottleneck_ch,
        groups=groups,
        stride=stride,
    )


class TableResidualNet(nn.Module):
    """MNIST CNN using standard residual blocks."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(1, 32, kernel_size=3, stride=1, padding=1),
            ConvBNAct(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=1),
            ConvBNAct(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(256, 256, stride=1),
        )
        self.head = LinearHead(256, num_classes=num_classes, dropout=dropout)
        self.out_channels = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)


class TableInceptionNet(nn.Module):
    """MNIST CNN using Inception-style blocks."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.0) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(1, 32, kernel_size=3, stride=1, padding=1),
            ConvBNAct(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.blocks = nn.Sequential(
            InceptionFactorizedBlock(64, 16, 16, 16, 16, 16, 16),
            InceptionFactorizedBlock(64, 32, 32, 32, 32, 32, 32),
            ConvBNAct(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            InceptionFactorizedBlock(256, 64, 64, 64, 64, 64, 64),
        )
        self.head = LinearHead(256, num_classes=num_classes, dropout=dropout)
        self.out_channels = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)


class TableResNeXtNet(nn.Module):
    """MNIST CNN using ResNeXt-style bottleneck blocks."""

    def __init__(self, num_classes: int = 10, groups: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(1, 32, kernel_size=3, stride=1, padding=1),
            ConvBNAct(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.blocks = nn.Sequential(
            make_resnext_block(64, 64, bottleneck_ch=32, groups=groups),
            make_resnext_block(64, 128, bottleneck_ch=64, groups=groups),
            ConvBNAct(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            make_resnext_block(256, 256, bottleneck_ch=128, groups=groups),
        )
        self.head = LinearHead(256, num_classes=num_classes, dropout=dropout)
        self.out_channels = num_classes
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)


class MNISTResInception(nn.Module):
    """Custom model combining residual blocks and an Inception block."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(1, 32, kernel_size=3),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
        )
        self.inception = InceptionFactorizedBlock(
            in_ch=64,
            pool_proj=16,
            out_1x1=16,
            reduce_3x3=16,
            out_3x3=32,
            reduce_double_3x3=16,
            out_double_3x3=32,
            stride=1,
        )
        self.tail = nn.Sequential(
            ConvBNAct(96, 128, kernel_size=1, padding=0),
            ResidualBlock(128, 128, stride=2),
            ResidualBlock(128, 128),
        )
        self.head = LinearHead(128, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.tail(self.inception(self.stem(x))))


MODEL_REGISTRY = {
    "residual": TableResidualNet,
    "inception": TableInceptionNet,
    "resnext": TableResNeXtNet,
    "custom": MNISTResInception,
}


def build_model(
    model_name: str,
    num_classes: int = 10,
    dropout: float = 0.0,
    groups: int = 4,
) -> nn.Module:
    """Build a model from a short registry name."""

    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(MODEL_REGISTRY)}.")

    if key == "resnext":
        return TableResNeXtNet(num_classes=num_classes, groups=groups, dropout=dropout)
    if key == "custom":
        # The custom model has its own default dropout, but accepts overrides.
        return MNISTResInception(num_classes=num_classes, dropout=dropout)
    return MODEL_REGISTRY[key](num_classes=num_classes, dropout=dropout)
