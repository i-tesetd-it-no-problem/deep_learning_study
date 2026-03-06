# PyTorch 迁移学习进阶模板（自动下载数据版）

这是一个适合入门和练手的 **PyTorch 迁移学习项目模板**。

相比上一版，这一版做了几个关键优化：

1. **默认自动下载 CIFAR-10 数据集**
2. **不再要求你手动准备 ImageFolder 格式数据**
3. **保留两阶段迁移学习流程**
4. **保留数据增强、学习率调度器、early stopping、混淆矩阵、分类报告**
5. **代码里加入了大量中文注释，适合边看边学**

---

## 1. 这套工程适合什么人

适合：

- 已经学完深度学习基础
- 想真正看懂迁移学习代码长什么样
- 手里还没有自己的数据集
- 想先拿一个公开数据集把迁移学习流程跑通

---

## 2. 这套工程默认做什么任务

默认任务：

> **使用 ResNet18 预训练模型，在 CIFAR-10 上做迁移学习分类**

CIFAR-10 一共有 10 个类别：

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

## 3. 迁移学习流程是什么

这套代码分两个阶段训练：

### 阶段 1：只训练分类头

- 预训练主干全部冻结
- 只训练最后新加的 `fc` 层
- 目标是先让新分类层快速适配 CIFAR-10

### 阶段 2：解冻 `layer4 + fc` 微调

- 解冻 ResNet18 最后一个 stage
- 继续用更小学习率训练
- 目标是让高层特征更贴合当前任务

这就是非常典型的迁移学习工程套路。

---

## 4. 目录结构

```text
transfer_learning_auto_dataset_template/
├─ dataset.py
├─ model.py
├─ train.py
├─ infer.py
├─ utils.py
├─ requirements.txt
└─ README.md
```

文件作用：

- `dataset.py`：自动下载 CIFAR-10，构建 train/val/test DataLoader
- `model.py`：加载预训练 ResNet18，替换 head，冻结/解冻层
- `train.py`：训练、验证、调度器、early stopping、保存最优模型
- `infer.py`：加载训练好的模型做预测
- `utils.py`：随机种子、画图、混淆矩阵、分类报告等工具函数

---

## 5. 环境准备

建议先在 venv 中安装依赖。

### 如果你用 CPU 版 PyTorch

```bash
pip install -r requirements.txt
```

### 如果你用 GPU 版 PyTorch

不要直接 `pip install -r requirements.txt` 来装 torch，建议先单独装 CUDA 版：

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install matplotlib scikit-learn pillow numpy
```

> 注意：不要把 `-i 清华源` 和 `--index-url https://download.pytorch.org/whl/cu128` 混着用，否则很容易又装成 CPU 版。

---

## 6. 直接开始训练

第一次运行时会自动下载 CIFAR-10：

```bash
python train.py
```

如果检测到 GPU，程序会自动使用 GPU。

你会在终端里看到类似输出：

- 当前使用设备: cuda
- GPU 名称: NVIDIA GeForce RTX 4070 ...

如果没有 GPU，就会自动退回 CPU。

---

## 7. 训练完成后会生成什么

默认输出目录：

```text
outputs/
├─ class_names.json
├─ checkpoints/
│  └─ best_model.pth
└─ reports/
   ├─ acc_curve.png
   ├─ loss_curve.png
   ├─ history.json
   ├─ test_classification_report.txt
   └─ test_confusion_matrix.png
```

含义：

- `best_model.pth`：验证集表现最好的模型权重
- `class_names.json`：类别名映射
- `acc_curve.png`：准确率曲线
- `loss_curve.png`：损失曲线
- `test_classification_report.txt`：测试集分类报告
- `test_confusion_matrix.png`：测试集混淆矩阵

---

## 8. 训练完成后如何预测

### 方式 1：直接从 CIFAR-10 测试集拿一张图做演示

```bash
python infer.py
```

也可以指定测试集里的第几张图：

```bash
python infer.py --sample-index 25
```

### 方式 2：预测你自己的图片

```bash
python infer.py --image ./your_image.jpg
```

---

## 9. 你最应该重点看哪些代码

建议按这个顺序看：

### 第一优先级

- `model.py`
- `train.py`

先理解：

- 预训练模型怎么加载
- 最后一层怎么替换
- 为什么先冻结再解冻
- 优化器只更新哪些参数

### 第二优先级

- `dataset.py`
- `utils.py`

再理解：

- 自动下载数据集是怎么做的
- train/val 是怎么划分的
- 混淆矩阵和分类报告怎么生成

### 第三优先级

- `infer.py`

再理解：

- 推理时怎么加载模型
- 为什么要和训练时保持相同预处理

---

## 10. 这套模板为什么有学习价值

因为它基本覆盖了一个真实小项目里最常见的核心环节：

- 自动拿到数据
- 构建 DataLoader
- 使用预训练模型
- 替换分类头
- 两阶段迁移学习
- 训练 / 验证 / 测试分离
- 学习率调度
- early stopping
- 保存 best model
- 输出混淆矩阵和分类报告
- 单独推理脚本

你把这套工程吃透后，再去看更复杂的：

- EfficientNet
- ViT
- BERT
- Wav2Vec2

本质上都会更容易看懂。

---

## 11. 最推荐你的学习方式

建议你这样学：

1. 先直接跑通 `python train.py`
2. 看终端日志，理解训练阶段 1 和阶段 2
3. 打开 `outputs/reports/` 看曲线、分类报告、混淆矩阵
4. 再回头逐行阅读 `model.py` 和 `train.py`
5. 自己尝试改：
   - epoch 数
   - batch size
   - 学习率
   - 是否只训练 head
   - 是否改成 ResNet50

这样你会比只看概念更快进入真实项目状态。
