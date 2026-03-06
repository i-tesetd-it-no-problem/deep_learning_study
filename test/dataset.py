from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DatasetBundle:
    # 保存和数据相关的所有对象，避免返回一长串零散变量
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_size: int
    val_size: int
    test_size: int
    class_names: List[str]


class TransformSubset(Dataset):
    # random_split 之后拿到的是 Subset
    # 但训练集和验证集通常需要不同的 transform
    # 所以这里用一个简单的包装器，把“取样本”和“应用 transform”分开
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int = 224):
    # 训练时使用更强一些的数据增强
    # 这样可以提升模型鲁棒性，减轻过拟合
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # 验证和测试阶段不应该再做随机增强
    # 否则每次评估结果都会有抖动，不利于稳定比较实验
    eval_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def build_cifar10_dataloaders(
    data_root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    image_size: int = 224,
    seed: int = 42,
) -> DatasetBundle:
    # 这个函数会自动下载 CIFAR-10 数据集
    # 这是当前工程最核心的改动之一：
    # 你不再需要手动准备 ImageFolder 格式的数据目录
    # 第一次运行会自动下载，后续运行会直接复用本地缓存

    train_transform, eval_transform = build_transforms(image_size=image_size)

    # 先不加 transform，把原始 PIL Image 取出来
    # 这样我们后面可以对训练 / 验证子集分别套不同的 transform
    full_train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=None,
    )

    test_base_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=None,
    )

    total_train_size = len(full_train_dataset)
    val_size = int(total_train_size * val_ratio)
    train_size = total_train_size - val_size

    # 手动设置随机种子，保证每次 train/val 划分一致
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_dataset,
        lengths=[train_size, val_size],
        generator=generator,
    )

    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset = TransformSubset(val_subset, transform=eval_transform)
    test_dataset = TransformSubset(test_base_dataset, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_size=train_size,
        val_size=val_size,
        test_size=len(test_dataset),
        class_names=full_train_dataset.classes,
    )
