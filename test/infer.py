from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



def build_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])



def load_class_names(class_path: str) -> List[str]:
    with open(class_path, "r", encoding="utf-8") as f:
        return json.load(f)



def build_model(weight_path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_tensor(model: nn.Module, image_tensor: torch.Tensor, class_names: List[str]) -> Tuple[str, float]:
    outputs = model(image_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    pred_name = class_names[pred_idx]
    pred_prob = probs[0, pred_idx].item()
    return pred_name, pred_prob



def main() -> None:
    parser = argparse.ArgumentParser(description="使用训练好的迁移学习模型做预测")
    parser.add_argument("--weights", default="./outputs/checkpoints/best_model.pth")
    parser.add_argument("--classes", default="./outputs/class_names.json")
    parser.add_argument("--image", default=None, help="可选：传入一张外部图片路径")
    parser.add_argument("--data-root", default="./data", help="如果不传 image，就自动从 CIFAR-10 测试集取样本")
    parser.add_argument("--sample-index", type=int, default=0, help="从 CIFAR-10 测试集选第几个样本做演示")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(args.classes)
    model = build_model(args.weights, num_classes=len(class_names), device=device)
    transform = build_eval_transform(image_size=args.image_size)

    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
        x = transform(image).unsqueeze(0).to(device)
        pred_name, pred_prob = predict_tensor(model, x, class_names)
        print(f"预测类别: {pred_name}")
        print(f"预测概率: {pred_prob:.4f}")
        return

    # 默认模式：直接从自动下载好的 CIFAR-10 测试集取一张样本预测
    test_dataset = datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=None,
    )
    image, label = test_dataset[args.sample_index]
    x = transform(image).unsqueeze(0).to(device)
    pred_name, pred_prob = predict_tensor(model, x, class_names)

    print(f"样本索引: {args.sample_index}")
    print(f"真实类别: {class_names[label]}")
    print(f"预测类别: {pred_name}")
    print(f"预测概率: {pred_prob:.4f}")


if __name__ == "__main__":
    main()
