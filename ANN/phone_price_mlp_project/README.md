# Phone Price MLP Project

一个更接近工程化的大项目模板，基于 PyTorch + CSV 表格数据 + MLP/ANN 多分类任务。

## 项目特点

- 配置文件管理（YAML）
- `src/` 工程化拆分
- 训练 / 验证 / 测试完整闭环
- 保存最佳模型、标准化器、标签映射、特征名
- 支持推理脚本
- 输出分类报告、混淆矩阵、训练曲线、学习率曲线
- 支持 Early Stopping / ReduceLROnPlateau / AMP / Gradient Clipping
- 基础日志与目录管理

## 目录结构

```text
phone_price_mlp_project/
├── configs/
│   └── phone_price_mlp.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── src/
│   ├── data/
│   │   └── tabular_data.py
│   ├── engine/
│   │   ├── evaluator.py
│   │   └── trainer.py
│   ├── models/
│   │   └── mlp.py
│   └── utils/
│       ├── checkpoint.py
│       ├── config.py
│       ├── logger.py
│       ├── plotting.py
│       └── seed.py
├── outputs/
├── requirements.txt
└── README.md
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据要求

默认假设 CSV:

- 最后一列是标签
- 其余列是数值特征
- 标签可以是整数类别，或字符串类别

默认配置文件中数据路径为：

```yaml
data:
  csv_path: ./手机价格预测.csv
```

把你的 CSV 放到项目根目录，或者修改 YAML 里的路径。

## 训练

```bash
python scripts/train.py --config configs/phone_price_mlp.yaml
```

训练完成后，会在 `outputs/exp_name_timestamp/` 下生成：

- `best_model.pt`
- `artifacts.joblib`
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `history.png`
- `lr_curve.png`
- `train.log`

## 单独评估

```bash
python scripts/evaluate.py \
  --config configs/phone_price_mlp.yaml \
  --run-dir outputs/your_run_dir
```

## 单样本 / 批量推理

### 单样本 JSON

```bash
python scripts/predict.py \
  --run-dir outputs/your_run_dir \
  --input-json '{"battery_power": 842, "blue": 0, "clock_speed": 2.2}'
```

注意：JSON 里的字段名必须和训练时特征列名一致。

### CSV 推理

```bash
python scripts/predict.py \
  --run-dir outputs/your_run_dir \
  --input-csv ./new_samples.csv \
  --save-path ./predictions.csv
```

## 适用范围

这个工程适合：

- 表格数据多分类项目
- PyTorch MLP/ANN 项目基线
- 中小型项目原型
- 从脚本版升级到项目版的训练工程

## 进一步升级建议

如果你后面真要继续往更大型工程走，可以继续补：

- KFold / StratifiedKFold
- 类别不平衡处理
- 多模型注册机制
- TensorBoard / WandB / MLflow
- 更完善的异常处理
- 单元测试
- CI/CD
- Docker
- 分布式训练
