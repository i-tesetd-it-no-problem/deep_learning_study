from __future__ import annotations

from typing import List

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# 这里默认使用 ResNet18 作为迁移学习 backbone
# 原因：
# 1. 结构经典，教程多，容易理解
# 2. 参数量比 ResNet50 小，对入门更友好
# 3. 在 CIFAR-10 这种小数据集上足够作为练手模板


def build_transfer_model(num_classes: int) -> nn.Module:
    # 加载预训练权重，而不是随机初始化
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # 第一阶段先冻结全部参数
    # 这样模型主体只负责“提特征”，我们先让最后分类头快速适配新任务
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model



def unfreeze_layer4_and_fc(model: nn.Module) -> nn.Module:
    # 第二阶段只解冻最后一个 stage
    # 这是迁移学习中很常见的折中方案：
    # 不会像全模型微调那样太激进，又比只训练头部更有适应能力
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    return model



def get_trainable_parameter_names(model: nn.Module) -> List[str]:
    names: List[str] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
    return names
