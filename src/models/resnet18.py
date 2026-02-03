"""ResNet-18 wrapper for CIFAR-10."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


def build_resnet18(num_classes: int = 10) -> nn.Module:
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
