from __future__ import annotations

import copy
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import build_cifar10_dataloaders
from model import build_transfer_model, get_trainable_parameter_names, unfreeze_layer4_and_fc
from utils import (
    EarlyStopping,
    ensure_dir,
    evaluate_model,
    save_classification_artifacts,
    save_history_plot,
    save_json,
    set_seed,
)


# =========================
# 这里集中放训练配置，便于你后续自己改
# =========================
DATA_ROOT = "./data"
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
IMAGE_SIZE = 224
VAL_RATIO = 0.1

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 5
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 3



def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 每个 batch 开始前都必须清空梯度
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        # 反向传播 + 参数更新
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels).item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels).item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc



def run_training_stage(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
    stage_name: str,
    num_epochs: int,
    history: Dict[str, List[float]],
    checkpoint_path: str,
) -> Tuple[nn.Module, float]:
    print(f"\n========== 开始 {stage_name} ==========")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        mode="max",
        min_delta=1e-4,
    )

    for epoch in range(num_epochs):
        print(f"[{stage_name}] Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # ReduceLROnPlateau 会根据验证集指标自动调低学习率
        # 这里监控 val_loss，因为 loss 往往比 acc 更平滑
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f}")
        print(f"val   loss: {val_loss:.4f} | val   acc: {val_acc:.4f} | lr: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_path)
            print(f"发现更优模型，已保存到: {checkpoint_path}")

        if early_stopping.step(val_acc):
            print("触发 early stopping，当前阶段提前结束。")
            break

        print("-" * 72)

    model.load_state_dict(best_model_wts)
    print(f"========== 结束 {stage_name}，最佳 val acc = {best_val_acc:.4f} ==========")
    return model, best_val_acc



def main() -> None:
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(CHECKPOINT_DIR)
    ensure_dir(REPORT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    else:
        print("当前未检测到 CUDA，将使用 CPU 训练。")

    # =========================
    # 1. 自动下载并构建 CIFAR-10 数据
    # =========================
    bundle = build_cifar10_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_ratio=VAL_RATIO,
        image_size=IMAGE_SIZE,
        seed=SEED,
    )

    print(f"训练集样本数: {bundle.train_size}")
    print(f"验证集样本数: {bundle.val_size}")
    print(f"测试集样本数: {bundle.test_size}")
    print(f"类别名称: {bundle.class_names}")

    save_json(bundle.class_names, os.path.join(OUTPUT_DIR, "class_names.json"))

    # =========================
    # 2. 构建迁移学习模型
    # =========================
    model = build_transfer_model(num_classes=len(bundle.class_names))
    model = model.to(device)

    print("第一阶段可训练参数:")
    print(get_trainable_parameter_names(model))

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # =========================
    # 3. 第一阶段：只训练分类头
    # =========================
    optimizer_stage1 = optim.AdamW(
        model.fc.parameters(),
        lr=STAGE1_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_stage1 = ReduceLROnPlateau(
        optimizer_stage1,
        mode="min",
        factor=0.5,
        patience=1,
    )

    model, best_acc_stage1 = run_training_stage(
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        criterion=criterion,
        optimizer=optimizer_stage1,
        scheduler=scheduler_stage1,
        device=device,
        stage_name="阶段1: 只训练分类头",
        num_epochs=STAGE1_EPOCHS,
        history=history,
        checkpoint_path=best_ckpt,
    )

    print(f"阶段1 最佳验证准确率: {best_acc_stage1:.4f}")

    # =========================
    # 4. 第二阶段：解冻 layer4 + fc 微调
    # =========================
    model = unfreeze_layer4_and_fc(model)
    print("第二阶段可训练参数:")
    print(get_trainable_parameter_names(model))

    optimizer_stage2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_stage2 = ReduceLROnPlateau(
        optimizer_stage2,
        mode="min",
        factor=0.5,
        patience=1,
    )

    model, best_acc_stage2 = run_training_stage(
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        criterion=criterion,
        optimizer=optimizer_stage2,
        scheduler=scheduler_stage2,
        device=device,
        stage_name="阶段2: 解冻 layer4 + fc 微调",
        num_epochs=STAGE2_EPOCHS,
        history=history,
        checkpoint_path=best_ckpt,
    )

    print(f"阶段2 最佳验证准确率: {best_acc_stage2:.4f}")

    # =========================
    # 5. 用最佳模型在测试集上做完整评估
    # =========================
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    y_true, y_pred = evaluate_model(model, bundle.test_loader, device)
    save_classification_artifacts(
        y_true=y_true,
        y_pred=y_pred,
        class_names=bundle.class_names,
        save_dir=REPORT_DIR,
        prefix="test",
    )

    save_history_plot(history, REPORT_DIR)
    save_json(history, os.path.join(REPORT_DIR, "history.json"))

    print("训练完成。已输出:")
    print(f"- 最优权重: {best_ckpt}")
    print(f"- 曲线与报告目录: {REPORT_DIR}")


if __name__ == "__main__":
    main()
