from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torchinfo import summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tabular_data import load_split_build_data
from src.engine.trainer import run_training
from src.models.mlp import TabularMLP
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP model for tabular multi-class classification")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(config["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['project']['exp_name']}_{timestamp}"
    run_dir = Path(config["project"]["output_root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(run_dir / "train.log")
    logger.info("项目名: %s", config["project"]["name"])
    logger.info("运行目录: %s", run_dir)
    logger.info("设备: %s", device)

    data_bundle, train_loader, val_loader, test_loader = load_split_build_data(config, device)
    logger.info(
        "输入维度: %d | 类别数: %d | 训练集: %d | 验证集: %d | 测试集: %d",
        data_bundle.input_dim,
        data_bundle.output_dim,
        data_bundle.train_size,
        data_bundle.val_size,
        data_bundle.test_size,
    )

    model = TabularMLP(
        input_dim=data_bundle.input_dim,
        output_dim=data_bundle.output_dim,
        hidden_dims=config["model"]["hidden_dims"],
        dropout=config["model"]["dropout"],
        use_bn=config["model"]["use_bn"],
    ).to(device)

    summary_str = str(
        summary(
            model,
            input_size=(config["data"]["batch_size"], data_bundle.input_dim),
            device=device.type,
            verbose=0,
        )
    )
    logger.info("模型摘要:\n%s", summary_str)

    (run_dir / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _, _, metrics = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        data_bundle=data_bundle,
        config=config,
        run_dir=run_dir,
        device=device,
        logger=logger,
    )

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("训练完成，关键指标已保存")


if __name__ == "__main__":
    main()
