import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
)
from torchinfo import summary


# =========================
# 0. 全局超参数配置
# =========================
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001
NUM_WORKERS = 2
VAL_RATIO = 0.1
SEED = 42
BEST_MODEL_PATH = "best_cifar10_cnn.pth"

# 自动选择设备
# 如果有 CUDA 就用 GPU
# 否则退回 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR10 的类别名称
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# =========================
# 1. 固定随机种子
# =========================
def set_seed(seed=42):
    # Python 自带随机数种子
    random.seed(seed)

    # numpy 随机数种子
    np.random.seed(seed)

    # PyTorch CPU 随机数种子
    torch.manual_seed(seed)

    # PyTorch GPU 随机数种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 让卷积等运算尽量可复现
    # 注意：这可能会稍微影响一点速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2. 准备数据集
# =========================
def create_datasets():
    # CIFAR10 常用均值和标准差
    # 这组值是统计出来的，用于 Normalize
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # 训练集的数据增强
    # RandomCrop: 随机裁剪，增强平移鲁棒性
    # RandomHorizontalFlip: 随机水平翻转，增强泛化
    # ToTensor: 把图片转成 Tensor，范围变成 [0, 1]
    # Normalize: 标准化，加快训练并提高稳定性
    train_transform = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean, std),
        ]
    )

    # 验证集 / 测试集一般不做数据增强，只做张量转换和标准化
    test_transform = Compose([ToTensor(), Normalize(mean, std)])

    # 先分别创建两个完整数据集对象
    # 一个用于训练增强
    # 一个用于验证集，不做增强
    full_train_dataset_for_train = CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )

    full_train_dataset_for_val = CIFAR10(
        root="./data", train=True, download=True, transform=test_transform
    )

    # 测试集
    test_dataset = CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    return full_train_dataset_for_train, full_train_dataset_for_val, test_dataset


# =========================
# 3. 划分训练集和验证集
# =========================
def split_train_val_dataset(
    train_dataset_for_train, train_dataset_for_val, val_ratio=0.1
):
    total_size = len(train_dataset_for_train)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    # 为了保证两套数据的划分索引一致，使用同一个随机种子
    generator = torch.Generator().manual_seed(SEED)

    # 先对“训练增强版数据集”做划分，得到训练索引和验证索引
    train_subset, val_subset_dummy = random_split(
        train_dataset_for_train, [train_size, val_size], generator=generator
    )

    # 再对“验证/测试预处理版数据集”按同样方式划分
    generator = torch.Generator().manual_seed(SEED)
    train_subset_dummy, val_subset = random_split(
        train_dataset_for_val, [train_size, val_size], generator=generator
    )

    # 真正返回：
    # 训练集用带增强的 train_subset
    # 验证集用不带增强的 val_subset
    return train_subset, val_subset


# =========================
# 4. 创建 DataLoader
# =========================
def create_dataloaders(train_dataset, val_dataset, test_dataset):
    # pin_memory=True:
    # 当使用 GPU 时，CPU 到 GPU 的拷贝会更高效
    # num_workers:
    # 开启子进程加快数据加载
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# =========================
# 5. 定义优化后的 CNN 模型
# =========================
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # -------------------------
        # 第一组卷积模块
        # -------------------------
        # padding=1 能保持 32x32 不变
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------
        # 第二组卷积模块
        # -------------------------
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------
        # 第三组卷积模块
        # -------------------------
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout 用于降低过拟合风险
        self.dropout = nn.Dropout(0.5)

        # 输入图像 32x32
        # 经过三次池化后:
        # 32 -> 16 -> 8 -> 4
        # 最终输出形状:
        # (batch_size, 128, 4, 4)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        # 第一组卷积
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # 第二组卷积
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # 第三组卷积
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        # 输出层
        # 注意：这里不加 softmax
        # 因为 CrossEntropyLoss 内部已经处理了
        x = self.output(x)
        return x


# =========================
# 6. 训练一个 epoch
# =========================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # non_blocking=True 配合 pin_memory=True 使用时更高效
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# =========================
# 7. 验证 / 测试函数
# =========================
def evaluate(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


# =========================
# 8. 训练总流程
# =========================
def train_model(model, train_loader, val_loader, device):
    # 多分类常用损失函数
    criterion = nn.CrossEntropyLoss()

    # Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 学习率调度器
    # 每 5 个 epoch 学习率变为原来的一半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 用于记录训练过程
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        end_time = time.time()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 保存验证集上最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            f" | Time: {end_time - start_time:.2f}s"
        )

    print(f"\n最佳验证集准确率: {best_val_acc:.4f}")
    print(f"最佳模型已保存到: {BEST_MODEL_PATH}")

    return history


# =========================
# 9. 绘制训练曲线
# =========================
def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================
# 10. 计算混淆矩阵
# =========================
def compute_confusion_matrix(y_true, y_pred, num_classes):
    # 自己手写一个简单混淆矩阵，避免额外依赖 sklearn
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm


# =========================
# 11. 绘制混淆矩阵
# =========================
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 在每个格子中写数值
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


# =========================
# 12. 反标准化
# =========================
def denormalize_image(img_tensor):
    # 用于可视化时把 Normalize 后的图片恢复到接近原图效果
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    img_tensor = img_tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return img_tensor


# =========================
# 13. 展示预测结果
# =========================
def show_predictions(model, test_loader, device, num_images=8):
    model.eval()

    images, labels = next(iter(test_loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)

    images = images[:num_images].cpu()
    labels = labels[:num_images].cpu()
    predicted = predicted[:num_images].cpu()

    plt.figure(figsize=(16, 4))

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)

        # 反标准化再显示
        img = denormalize_image(images[i]).permute(1, 2, 0).numpy()

        plt.imshow(img)
        plt.title(
            f"P:{CLASS_NAMES[predicted[i]]}\nT:{CLASS_NAMES[labels[i]]}", fontsize=9
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# 14. 主函数
# =========================
def main():
    set_seed(SEED)

    print(f"当前设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    else:
        print("当前未检测到可用 GPU，将使用 CPU")

    # 创建原始数据集
    full_train_dataset_for_train, full_train_dataset_for_val, test_dataset = (
        create_datasets()
    )

    # 划分训练集和验证集
    train_dataset, val_dataset = split_train_val_dataset(
        full_train_dataset_for_train, full_train_dataset_for_val, val_ratio=VAL_RATIO
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    # 创建 DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    # 创建模型
    model = ImageModel().to(DEVICE)

    # 打印模型结构摘要
    # device 要和模型所在设备一致
    summary(model, input_size=(BATCH_SIZE, 3, 32, 32), device=str(DEVICE))

    # 训练模型
    history = train_model(model, train_loader, val_loader, DEVICE)

    # 绘制训练曲线
    plot_training_history(history)

    # 加载验证集上表现最好的模型，再做最终测试
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, DEVICE
    )

    print(f"\n最终测试集 Loss: {test_loss:.4f}")
    print(f"最终测试集 Acc : {test_acc:.4f}")

    # 绘制混淆矩阵
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=10)
    plot_confusion_matrix(cm, CLASS_NAMES)

    # 展示部分预测结果
    show_predictions(model, test_loader, DEVICE, num_images=8)


if __name__ == "__main__":
    main()
