from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



def set_seed(seed: int = 42) -> None:
    # 固定随机种子，尽量让实验可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 下面这两项能进一步提高可复现性
    # 但有些场景可能会稍微影响性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def save_json(data, save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def save_history_plot(history: Dict[str, List[float]], save_dir: str) -> None:
    # 画出训练过程中的 loss / acc 曲线
    # 这样你能直观看到：
    # 1. 训练是否收敛
    # 2. 是否过拟合
    # 3. 第二阶段微调后是否继续提升
    ensure_dir(save_dir)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "acc_curve.png"), dpi=200)
    plt.close()



def evaluate_model(model, dataloader, device) -> Tuple[List[int], List[int]]:
    # 收集整套数据上的真实标签和预测标签
    # 后面生成混淆矩阵、分类报告都需要这些结果
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return y_true, y_pred



def save_classification_artifacts(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_dir: str,
    prefix: str = "test",
) -> None:
    ensure_dir(save_dir)

    # 分类报告会输出每一类的 precision / recall / f1-score
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
    with open(os.path.join(save_dir, f"{prefix}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    # 混淆矩阵非常适合做错误分析
    # 它能帮你看出“哪两个类最容易互相混淆”
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title(f"{prefix.capitalize()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_confusion_matrix.png"), dpi=220)
    plt.close(fig)


class EarlyStopping:
    # 提前停止机制
    # 如果验证指标连续若干轮都没有改善，就停止训练
    # 这样可以：
    # 1. 节省时间
    # 2. 减少无意义训练
    # 3. 降低过拟合风险
    def __init__(self, patience: int = 3, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def step(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
