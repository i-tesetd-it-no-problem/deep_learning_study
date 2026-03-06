"""
ANN案例的实现步骤
    1. 构建数据集
    2. 搭建神经网络
    3. 模型训练
    4. 模型测试

"""

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchinfo import summary
import time


# 1. 构建数据集
def create_dataset():
    # 1.1  加载CSV
    data = pd.read_csv("./手机价格预测.csv")  # (2000, 21) 20个特征 一个分类

    # 1.2 获取X特征列 和Y标签列
    x, y = data.iloc[:, :-1], data.iloc[:, -1]  # x:(2000, 20) y:(2000,)

    # 1.3 把特征列转为浮点值
    x = x.astype(np.float32)

    # 1.4 切分训练集和测试集
    # stratify=y: 按照y的分布切分训练集和测试集，保证训练集和测试集中y的分布相似
    # random_state=3: 固定随机种子，保证每次切分结果一致，便于复现
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=3, stratify=y
    )

    # 1.5 对数据进行标准化
    # 标准化是表格数据ANN中非常重要的一步
    # 作用:
    #   1. 消除不同特征之间量纲差异
    #   2. 避免某些数值特别大的特征主导训练
    #   3. 通常能让训练更稳定、收敛更快
    # 注意:
    #   1. 只能在训练集上fit
    #   2. 测试集只能用训练集得到的均值和方差做transform
    # 如果出现这些问题，可以优先检查这里:
    #   1. loss下降很慢
    #   2. 训练不稳定，波动很大
    #   3. 准确率长期上不去
    # 可选方案:
    #   1. StandardScaler 标准化(当前使用，最常见)
    #   2. MinMaxScaler 归一化到[0, 1]
    #   3. RobustScaler 对异常值更稳
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 1.6 把数据集封装成张量数据集 参数：特征张量，标签张量
    # 特征一般用float32
    # 多分类标签一般用long，因为CrossEntropyLoss要求类别标签是整数索引
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.long),
    )

    # 1.7 把测试集封装成张量数据集 参数：特征张量，标签张量
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.long),
    )

    # 1.8 返回结果 参数：训练集，测试集，输入特征数(20)，类别数(4)
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


# 2. 搭建神经网络
class PhonePriceModel(nn.Module):
    """
    param input_dim: 输入特征数
    param output_dim: 类别数
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim  # 输入特征数
        self.output_dim = output_dim  # 类别数

        # 隐藏层1 [20, 128] 隐藏层2 [128, 256] 输出层 [256, 4]
        # 当前是一个比较基础的MLP结构
        # 如果出现这些问题，可以优化这里:
        #   1. 模型太简单，训练准确率和测试准确率都不高 -> 可以增加网络深度/宽度
        #   2. 模型过拟合，训练准确率高但测试准确率低 -> 可以减小网络规模、加Dropout、加正则化
        # 可选优化方案:
        #   1. 更浅的网络: 20 -> 64 -> 4
        #   2. 更深的网络: 20 -> 128 -> 256 -> 128 -> 4
        #   3. 更宽的网络: 20 -> 256 -> 256 -> 4
        # 注意:
        #   1. 表格数据不一定越深越好
        #   2. 小数据集下网络过深更容易过拟合
        self.linear1 = nn.Linear(self.input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, self.output_dim)

        # Dropout层
        # 作用:
        #   1. 训练时随机丢弃一部分神经元，减少过拟合
        # 如果出现这些问题，可以考虑开启Dropout:
        #   1. 训练准确率很高，测试准确率明显偏低
        #   2. 训练loss持续下降，但测试效果不提升
        # 常见设置:
        #   1. 0.2
        #   2. 0.3
        #   3. 0.5
        # 当前先保留但默认启用一个中等强度的Dropout
        self.dropout = nn.Dropout(p=0.3)

        # BN(Batch Normalization)层
        # BN和输入标准化不是一回事:
        #   1. 输入标准化是对原始输入特征做处理
        #   2. BN是对网络中间层输出做处理
        # 如果出现这些问题，可以考虑BN:
        #   1. 训练不稳定
        #   2. loss震荡明显
        #   3. 学习率稍大时训练容易发散
        # 注意:
        #   1. 表格ANN不一定必须加BN
        #   2. 先做输入标准化通常更重要
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

    # 前向传播函数
    def forward(self, x):
        # 2.1 隐藏层1 加权求和
        x = self.linear1(x)

        # 2.2 BN归一化
        # 如果你想做消融实验，可以临时注释掉BN看看效果变化
        x = self.bn1(x)

        # 2.3 激活函数
        # 当前使用ReLU，最常见
        # 如果出现“神经元死亡”或者训练效果一般，可以试:
        #   1. LeakyReLU
        #   2. GELU
        x = torch.relu(x)

        # 2.4 Dropout
        x = self.dropout(x)

        # 2.5 隐藏层2 加权求和
        x = self.linear2(x)

        # 2.6 BN归一化
        x = self.bn2(x)

        # 2.7 激活函数
        x = torch.relu(x)

        # 2.8 Dropout
        x = self.dropout(x)

        # 2.9 输出层 加权求和
        # 后续用多分类交叉熵损失函数，所以输出层不需要激活函数
        # CrossEntropyLoss函数内部会自动调用softmax相关计算
        return self.output(x)


# 3. 模型训练
def train_model(train_dataset, model, device):
    # 1. 创建数据加载器 参数：训练集，批次大小，是否打乱数据
    # batch_size是一个重要超参数
    # 如果出现这些问题，可以优化这里:
    #   1. 显存不够 -> 减小batch_size
    #   2. 训练不够稳定 -> 可以尝试稍大一点batch_size
    #   3. 收敛速度慢 -> 可以结合硬件尝试更大的batch_size
    # 常见可选值:
    #   8 / 16 / 32 / 64
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 2. 定义损失函数，多分类交叉熵损失函数适用于多分类问题
    # 如果类别极度不平衡，可以考虑给CrossEntropyLoss加权重 weight=
    criterion = nn.CrossEntropyLoss()

    # 3. 定义优化器 参数：模型参数，学习率
    # 当前用Adam，适合大多数入门场景，收敛快，比较稳
    # 如果出现这些问题，可以优化这里:
    #   1. 收敛慢 -> Adam通常比SGD更快
    #   2. 泛化一般 -> 可以试SGD + momentum
    #   3. 过拟合 -> 可以试AdamW并加weight_decay
    # 可选方案:
    #   1. Adam
    #   2. SGD(momentum=0.9)
    #   3. AdamW(weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 学习率调度器
    # 作用:
    #   1. 前期学习快一点
    #   2. 后期学习慢一点，训练更稳定
    # 如果出现这些问题，可以优化这里:
    #   1. 前期下降正常，后期不再提升 -> 可以降低学习率
    #   2. loss在某个阶段震荡不收敛 -> 可以尝试学习率衰减
    # 当前使用StepLR作为示例:
    #   每30轮，学习率乘0.5
    # 可选方案:
    #   1. StepLR
    #   2. ReduceLROnPlateau
    #   3. CosineAnnealingLR
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 5. 定义列表记录每轮训练的平均损失和准确率
    # 为什么训练里也统计准确率:
    #   1. 训练准确率用于观察模型是否学会
    #   2. 测试准确率用于观察泛化能力
    # 训练acc和测试acc都应该看
    train_loss_list = []
    train_acc_list = []

    # 6. 模型训练
    # epochs也是一个重要超参数
    # 如果出现这些问题，可以优化这里:
    #   1. 训练还没收敛 -> 增加epochs
    #   2. 训练很早就不再提升 -> 可以减少epochs
    #   3. 训练集效果越来越好，测试集反而变差 -> 可能过拟合
    epochs = 100  # 训练轮数

    for i in range(epochs):
        # 6.1 切换模型状态为训练模式
        model.train()

        # 6.2 变量定义
        total_loss = 0.0  # 累积损失
        cur_batch_size = 0  # 当前轮累计样本数
        correct = 0  # 当前轮预测正确的样本数
        start_time = time.time()  # 记录开始时间

        for x, y in train_loader:
            # 6.3 把每个批次的数据移动到指定设备(GPU或CPU)
            x = x.to(device)
            y = y.to(device)

            # 6.4 梯度清零
            optimizer.zero_grad()

            # 6.5 模型预测
            y_pred = model(x)

            # 6.6 计算损失 参数：预测结果，真实标签
            loss = criterion(y_pred, y)

            # 6.7 反向传播
            loss.backward()

            # 6.8 更新参数
            optimizer.step()

            # 6.9 累积损失
            total_loss += loss.item() * x.size(0)

            # 6.10 累积样本数
            cur_batch_size += x.size(0)

            # 6.11 统计当前批次预测正确的样本数
            pred_label = torch.argmax(y_pred, dim=1)
            correct += (pred_label == y).sum().item()

        # 6.12 更新学习率
        scheduler.step()

        # 6.13 计算当前轮平均损失和准确率
        avg_loss = total_loss / cur_batch_size
        avg_acc = correct / cur_batch_size

        # 6.14 保存当前轮平均损失和准确率
        train_loss_list.append(avg_loss)
        train_acc_list.append(avg_acc)

        end_time = time.time()  # 记录结束时间
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {i + 1}/{epochs} 平均损失: {avg_loss:.4f} 训练准确率: {avg_acc:.4f} 学习率: {current_lr:.6f} 耗时: {end_time - start_time:.2f}秒"
        )

        # 6.15 这里是Early Stopping思路说明(当前未正式启用)
        # 如果出现这些问题，可以考虑早停:
        #   1. 训练轮数很多，但后期基本不再提升
        #   2. 测试集效果开始下降
        # 常见做法:
        #   1. 连续patience轮没有提升就停止训练
        #   2. 保存最佳模型参数
        # 当前代码没有引入验证集，所以这里只写思路，不直接启用
        # 如果后面要做更规范训练，建议:
        #   训练集 / 验证集 / 测试集 三划分

    # 6.16 模型训练完成，保存模型参数
    # 包含(权重矩阵和偏置矩阵)
    torch.save(model.state_dict(), "phone_price_model.pth")
    print("模型参数已保存到 phone_price_model.pth")

    # 6.17 返回训练过程中的损失和准确率
    return train_loss_list, train_acc_list


# 4. 模型测试
def test_evaluate(test_dataset, model, device):
    # 1. 创建测试集数据加载器 参数：测试集，批次大小，是否打乱数据
    # 测试时一般不打乱
    # batch_size可以和训练一致，也可以更大，只要显存允许
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 2. 切换模型状态为评估模式
    model.eval()

    # 3. 定义变量
    correct = 0  # 记录预测正确的样本数
    total_loss = 0.0  # 累积损失
    total_samples = 0  # 总样本数

    # 4. 定义损失函数，多分类交叉熵损失函数适用于多分类问题
    criterion = nn.CrossEntropyLoss()

    # 5. 关闭梯度计算，减少内存占用，加快测试速度
    with torch.no_grad():
        # 6. 从数据加载器中获取到每批次的数据
        for x, y in test_loader:
            # 6.1 把数据移动到指定设备
            x = x.to(device)
            y = y.to(device)

            # 6.2 模型预测
            y_pred = model(x)

            # 6.3 计算损失
            loss = criterion(y_pred, y)

            # 6.4 累积损失
            total_loss += loss.item() * x.size(0)

            # 6.5 根据加权求和得到类别 用argmax函数获取每行最大值的索引作为预测类别(替代了softmax函数，一样的效果)
            y_pred = torch.argmax(y_pred, dim=1)  # dim=1表示按行取最大值的索引

            # 6.6 统计预测正确的样本数
            correct += (y_pred == y).sum().item()

            # 6.7 累积总样本数
            total_samples += x.size(0)

    # 7. 模型测试完成，计算平均损失和准确率
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    print(f"模型测试完成，平均损失: {avg_loss:.4f} 准确率: {accuracy:.4f}")

    # 8. 对测试结果的解释
    # 如果出现这些情况，可以这样判断:
    #   1. 训练准确率高，测试准确率低 -> 可能过拟合
    #      可优化:
    #         - 加Dropout
    #         - 减小模型规模
    #         - 加weight_decay
    #         - 做早停
    #   2. 训练准确率低，测试准确率也低 -> 模型可能欠拟合
    #      可优化:
    #         - 增加网络深度/宽度
    #         - 增加训练轮数
    #         - 调大学习率或换优化器
    #   3. loss震荡、训练不稳 -> 可优化:
    #         - 降低学习率
    #         - 使用标准化
    #         - 使用BN
    #   4. 测试准确率长期上不去 -> 可优化:
    #         - 检查数据质量
    #         - 做特征工程
    #         - 调整模型结构
    #         - 调整优化器/学习率

    # 9. 返回测试损失和准确率
    return avg_loss, accuracy


# 5. 绘制训练曲线
def plot_history(train_loss_list, train_acc_list):
    # 1. 创建轮数列表
    epochs = range(1, len(train_loss_list) + 1)

    # 2. 创建画布
    plt.figure(figsize=(12, 5))

    # 3. 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # 4. 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_list, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()

    # 5. 调整布局并显示图像
    plt.tight_layout()
    plt.show()

    # 6. 如何看曲线
    # 如果出现这些情况，可以这样判断:
    #   1. loss持续下降，acc持续上升 -> 训练正常
    #   2. loss几乎不降 -> 学习率可能太小 / 模型太弱 / 数据问题
    #   3. loss大幅震荡 -> 学习率可能太大 / 训练不稳定
    #   4. 训练acc很高，但测试acc一般 -> 可能过拟合
    #   5. 很早就进入平台期 -> 可以尝试学习率衰减 / 调整模型 / 延长训练或早停


if __name__ == "__main__":
    # 1. 构建 训练集和测试集 输入特征数 类别数
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    print(f"输入特征数: {input_dim} 类别数: {output_dim}")

    # 2. 选择设备
    # 如果GPU可用，则优先使用GPU
    # 如果训练时报device不一致错误，通常要检查:
    #   1. 模型是否.to(device)
    #   2. 输入x是否.to(device)
    #   3. 标签y是否.to(device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前设备: {device}")

    # 3. 搭建神经网络
    model = PhonePriceModel(input_dim, output_dim).to(device)

    # 4. 打印模型结构和参数量
    summary(model, input_size=(16, input_dim), device=device)

    # 5. 模型训练
    train_loss_list, train_acc_list = train_model(train_dataset, model, device)

    # 6. 模型测试
    test_evaluate(test_dataset, model, device)

    # 7. 绘制训练曲线
    plot_history(train_loss_list, train_acc_list)

    # 8. 进一步可选优化总结
    # 当前代码已经包含/体现的优化点:
    #   1. 数据标准化
    #   2. Adam优化器
    #   3. 学习率调度器
    #   4. BatchNorm
    #   5. Dropout
    #   6. 训练准确率监控
    #   7. 测试损失和测试准确率监控
    #
    # 还可以进一步尝试的优化:
    #   1. 把Adam改成AdamW，并加weight_decay
    #      适用于: 训练集很好，测试集一般，怀疑过拟合
    #   2. 使用SGD + momentum
    #      适用于: 想比较不同优化器的泛化效果
    #   3. 调整batch_size
    #      适用于: 显存足够 / 训练稳定性一般 / 收敛速度一般
    #   4. 调整隐藏层结构
    #      适用于: 模型欠拟合或过拟合
    #   5. 使用验证集 + Early Stopping
    #      适用于: 想更规范地防止过拟合
    #   6. 做更好的特征工程
    #      适用于: 表格任务准确率长期卡住
