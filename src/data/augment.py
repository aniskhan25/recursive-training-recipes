"""Weak/strong augmentations for SSL."""

from __future__ import annotations

from typing import Tuple

from torchvision import transforms


def mnist_weak() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])


def mnist_strong() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomRotation(25),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])


def cifar_weak() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])


def cifar_strong() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
    ])
