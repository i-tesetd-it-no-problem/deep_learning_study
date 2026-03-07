"""
ANN案例的实现步骤
    1. 构建数据集
    2. 搭建神经网络
    3. 模型训练
    4. 模型验证
    5. 模型测试
    6. 训练曲线与混淆矩阵可视化

当前版本是一个比较完整的表格数据多分类ANN/MLP训练模板
适用于:
    1. CSV表格数据
    2. 多分类任务
    3. PyTorch入门到进阶的标准训练流程

当前代码包含的优化点:
    1. train / val / test 三划分
    2. 标准化(StandardScaler)
    3. AdamW优化器
    4. weight_decay正则化
    5. 学习率调度器 ReduceLROnPlateau
    6. Early Stopping
    7. BatchNorm
    8. Dropout
    9. GPU训练
    10. AMP混合精度训练
    11. 梯度裁剪
    12. 保存最佳模型
    13. 四张训练/验证曲线图
    14. 测试集最终评估
    15. 混淆矩阵和分类报告
"""

import os
import copy
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

from torchinfo import summary

# =========================
# 0. 全局配置
# =========================
SEED = 42
CSV_PATH = "./手机价格预测.csv"
BEST_MODEL_PATH = "best_phone_price_model.pth"

CONFIG = {
    # 数据划分
    "test_size": 0.2,  # 测试集占比
    "val_size": 0.2,  # 从训练验证集合中再切出验证集占比
    # DataLoader
    "batch_size": 32,  # 批次
    "num_workers": 0,  # Windows下建议先用0，最稳
    # 训练超参数
    "epochs": 200,  # 训练轮数
    "lr": 1e-3,  # 学习率
    "weight_decay": 1e-4,  # 权重衰减
    "gradient_clip": 1.0,  # 梯度裁剪阈值，None表示不裁剪
    # 学习率调度器
    "scheduler_factor": 0.5,  # 当验证集loss不下降时，学习率降低的倍数，例如0.5表示降低到原来的一半
    "scheduler_patience": 8,  # 学习率调度器的耐心值，即验证集loss多少轮不下降时触发学习率降低
    "scheduler_min_lr": 1e-6,  # 学习率调度器的最小学习率，防止学习率降低过多
    # Early Stopping
    "early_stopping_patience": 20,  # 早停的耐心值，即验证集loss多少轮不改善时触发早停
    "early_stopping_min_delta": 1e-4,  # 早停的最小改善值，只有当验证集loss改善超过这个值时才算真正提升，防止过早触发早停
    # 模型结构
    "hidden_dims": [
        128,
        256,
        128,
    ],  # 隐藏层结构，可以根据需要调整，例如 [64, 128] 或 [256, 256, 256]
    "dropout": 0.3,  # Dropout概率，0.3表示每个神经元有30%的概率被丢弃
    "use_bn": True,  # 是否使用BatchNorm，通常在MLP中使用BatchNorm可以加速训练和提升性能
    # 是否开启AMP混合精度训练，只有在GPU可用时才会启用
    "use_amp": True,
}


# =========================
# 1. 固定随机种子
# =========================
def set_seed(seed: int = 42):
    # 固定Python随机种子
    random.seed(seed)

    # 固定numpy随机种子
    np.random.seed(seed)

    # 固定torch随机种子
    torch.manual_seed(seed)

    # 固定cuda随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 为了尽量复现结果
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2. 构建数据集
# =========================
def create_datasets():
    # 2.1 加载CSV
    data = pd.read_csv(CSV_PATH)  # 形状通常为 (2000, 21)，20个特征 + 1个标签

    # 2.2 获取X特征列 和Y标签列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]  # x:(N, 20) y:(N,)

    # 2.3 把特征列转为浮点值
    # 表格数据喂给神经网络时，特征一般用float32
    x = x.astype(np.float32)

    # 2.4 第一次切分: 先切出测试集
    # stratify=y: 按类别分层抽样，保证各数据集类别分布相似
    # random_state=SEED: 固定随机种子，保证可复现
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=CONFIG["test_size"], random_state=SEED, stratify=y
    )

    # 2.5 第二次切分: 从训练验证集合中切出验证集
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=CONFIG["val_size"],
        random_state=SEED,
        stratify=y_train_val,
    )

    # 2.6 对数据进行标准化
    # 注意:
    #   1. 只能在训练集上fit
    #   2. 验证集和测试集只能使用训练集得到的均值和标准差进行transform
    # 这样可以避免数据泄漏
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # 2.7 把数据封装成张量数据集
    # 多分类标签一般用long，因为CrossEntropyLoss要求类别标签是整数索引
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.long),
    )

    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.long),
    )

    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.long),
    )

    # 2.8 返回结果
    input_dim = x_train.shape[1]
    output_dim = len(np.unique(y))

    meta_info = {
        "input_dim": input_dim,  # 输入特征数，例如20
        "output_dim": output_dim,  # 输出类别数，例如4
        "scaler_mean": scaler.mean_,  # scaler.scale_: 标准化的均值和标准差，可以在推理阶段使用同样的标准化参数对新数据进行处理，保证训练和推理的一致性
        "scaler_scale": scaler.scale_,  # 如果需要在推理阶段对新数据进行标准化处理，必须使用训练集上fit得到的均值和标准差，而不能重新计算，否则会导致数据泄漏和性能下降
        "train_size": len(train_dataset),  # 训练集样本数，例如1280
        "val_size": len(val_dataset),  # 验证集样本数，例如320
        "test_size": len(test_dataset),  # 测试集样本数，例如400
    }

    return train_dataset, val_dataset, test_dataset, meta_info


# =========================
# 3. 创建DataLoader
# =========================
def create_dataloaders(train_dataset, val_dataset, test_dataset, device):
    # pin_memory在GPU训练时通常有助于加速数据搬运
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


# =========================
# 4. 搭建神经网络
# =========================
class PhonePriceModel(nn.Module):
    """
    param input_dim: 输入特征数
    param output_dim: 类别数
    param hidden_dims: 隐藏层结构，例如 [128, 256, 128]
    param dropout: Dropout概率
    param use_bn: 是否使用BatchNorm
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims=None,
        dropout: float = 0.3,
        use_bn: bool = True,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 128]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout
        self.use_bn = use_bn

        layers = []
        prev_dim = input_dim

        # 4.1 动态构建MLP结构
        # 常见顺序:
        #   Linear -> BatchNorm -> ReLU -> Dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # 4.2 特征提取部分
        self.feature_extractor = nn.Sequential(*layers)

        # 4.3 输出层
        # 多分类任务 + CrossEntropyLoss
        # 输出层不需要手动加softmax
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # 4.4 初始化参数
        self._init_weights()

    def _init_weights(self):
        # 对线性层做Kaiming初始化，更适合ReLU类激活函数
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        return x


# =========================
# 5. 早停机制
# =========================
class EarlyStopping:
    """
    早停机制:
        1. 如果验证集loss连续若干轮没有改善，则停止训练
        2. 防止过拟合
        3. 节省训练时间
    """

    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss):
        # 当前loss比历史最好loss改善了至少min_delta，才算真正提升
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True


# =========================
# 6. 计算准确率
# =========================
def calculate_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total


# =========================
# 7. 单轮训练
# =========================
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp):
    # 切换到训练模式
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in train_loader:
        # 7.1 数据移动到指定设备
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # 7.2 梯度清零
        optimizer.zero_grad()

        # 7.3 AMP混合精度训练
        # 只有在cuda可用且use_amp=True时启用
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        # 7.4 反向传播
        scaler.scale(loss).backward()

        # 7.5 梯度裁剪
        # 如果训练不稳定、梯度爆炸，可以考虑这里
        if CONFIG["gradient_clip"] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])

        # 7.6 更新参数
        scaler.step(optimizer)
        scaler.update()

        # 7.7 统计loss和accuracy
        total_loss += loss.item() * x.size(0)
        correct, batch_size = calculate_accuracy(logits, y)
        total_correct += correct
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# =========================
# 8. 单轮验证/测试
# =========================
def evaluate_one_epoch(model, data_loader, criterion, device):
    # 切换到评估模式
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    # 验证和测试阶段不需要梯度
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            correct, batch_size = calculate_accuracy(logits, y)
            total_correct += correct
            total_samples += batch_size

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, all_preds, all_labels


# =========================
# 9. 完整训练流程
# =========================
def train_model(train_loader, val_loader, model, device):
    # 9.1 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 9.2 定义优化器
    # AdamW通常比Adam更适合带weight_decay的场景
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # 9.3 定义学习率调度器
    # ReduceLROnPlateau: 当验证集loss长期不下降时，自动降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"],
        min_lr=CONFIG["scheduler_min_lr"],
    )

    # 9.4 Early Stopping
    early_stopping = EarlyStopping(
        patience=CONFIG["early_stopping_patience"],
        min_delta=CONFIG["early_stopping_min_delta"],
    )

    # 9.5 AMP缩放器
    use_amp = CONFIG["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 9.6 记录训练历史
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_model_state = None
    best_val_loss = float("inf")
    best_epoch = 0

    # 9.7 开始训练
    for epoch in range(CONFIG["epochs"]):
        start_time = time.time()

        # 训练一轮
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )

        # 验证一轮
        val_loss, val_acc, _, _ = evaluate_one_epoch(
            model, val_loader, criterion, device
        )

        # 更新学习率调度器
        scheduler.step(val_loss)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        # 保存历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state_dict": best_model_state,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "input_dim": model.input_dim,
                    "output_dim": model.output_dim,
                    "hidden_dims": model.hidden_dims,
                    "dropout": model.dropout_p,
                    "use_bn": model.use_bn,
                    "config": CONFIG,
                },
                BEST_MODEL_PATH,
            )

        # Early Stopping检查
        early_stopping.step(val_loss)

        end_time = time.time()

        print(
            f"Epoch {epoch + 1}/{CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {end_time - start_time:.2f}s"
        )

        if early_stopping.should_stop:
            print(f"Early Stopping触发，提前结束训练，停止于第 {epoch + 1} 轮")
            break

    # 9.8 加载最佳模型参数
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"最佳验证集Loss: {best_val_loss:.4f}，对应Epoch: {best_epoch}")
    print(f"最佳模型已保存到 {BEST_MODEL_PATH}")

    return model, history


# =========================
# 10. 测试集评估
# =========================
def test_evaluate(test_loader, model, device):
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, preds, labels = evaluate_one_epoch(
        model, test_loader, criterion, device
    )

    print(f"模型测试完成，平均损失: {test_loss:.4f} 准确率: {test_acc:.4f}")

    return test_loss, test_acc, preds, labels


# =========================
# 11. 绘制四张训练/验证曲线图
# =========================
def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 10))

    # 11.1 训练损失
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.legend()

    # 11.2 验证损失
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    # 11.3 训练准确率
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train Accuracy")
    plt.legend()

    # 11.4 验证准确率
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# =========================
# 12. 绘制学习率曲线
# =========================
def plot_lr(history):
    epochs = range(1, len(history["lr"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["lr"], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# 13. 绘制混淆矩阵
# =========================
def plot_confusion_matrix(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()


def main():
    # 14.1 固定随机种子
    set_seed(SEED)

    # 14.2 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    # 14.3 构建数据集
    train_dataset, val_dataset, test_dataset, meta_info = create_datasets()

    print(
        f"输入特征数: {meta_info['input_dim']} | "
        f"类别数: {meta_info['output_dim']} | "
        f"训练集: {meta_info['train_size']} | "
        f"验证集: {meta_info['val_size']} | "
        f"测试集: {meta_info['test_size']}"
    )

    # 14.4 构建DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, device
    )

    # 14.5 搭建神经网络
    model = PhonePriceModel(
        input_dim=meta_info["input_dim"],
        output_dim=meta_info["output_dim"],
        hidden_dims=CONFIG["hidden_dims"],
        dropout=CONFIG["dropout"],
        use_bn=CONFIG["use_bn"],
    ).to(device)

    # 14.6 打印模型结构和参数量
    summary(
        model,
        input_size=(CONFIG["batch_size"], meta_info["input_dim"]),
        device=device.type,
    )

    # 14.7 模型训练
    model, history = train_model(train_loader, val_loader, model, device)

    # 14.8 模型测试
    test_loss, test_acc, preds, labels = test_evaluate(test_loader, model, device)

    # 14.9 绘制四张训练/验证曲线图
    plot_history(history)

    # 14.10 绘制学习率曲线
    plot_lr(history)

    # 14.11 绘制混淆矩阵
    plot_confusion_matrix(labels, preds, meta_info["output_dim"])

    # 14.12 输出分类报告
    print("分类报告:")
    print(classification_report(labels, preds, digits=4))


# =========================
# 14. 主函数
# =========================
if __name__ == "__main__":
    main()
