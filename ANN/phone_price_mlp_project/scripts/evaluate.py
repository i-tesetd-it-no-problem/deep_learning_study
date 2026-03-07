from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tabular_data import load_split_build_data
from src.engine.evaluator import evaluate_one_epoch
from src.models.mlp import TabularMLP
from src.utils.checkpoint import load_artifacts, load_checkpoint
from src.utils.config import load_yaml_config
from src.utils.logger import get_logger
from src.utils.plotting import save_confusion_matrix_plot
from src.utils.seed import set_seed



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件")
    parser.add_argument("--run-dir", type=str, required=True, help="训练输出目录")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_seed(config["project"]["seed"])

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir 不存在: {run_dir}")

    logger = get_logger(run_dir / "evaluate.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_bundle, _, _, test_loader = load_split_build_data(config, device)
    checkpoint = load_checkpoint(run_dir / "best_model.pt", map_location=device)
    artifacts = load_artifacts(run_dir / "artifacts.joblib")

    model = TabularMLP(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
        use_bn=checkpoint["use_bn"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate_one_epoch(model, test_loader, criterion, device)
    report_text = classification_report(
        labels,
        preds,
        target_names=artifacts["class_names"],
        digits=config["eval"].get("digits", 4),
        zero_division=0,
    )

    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    (run_dir / "re_eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "re_eval_classification_report.txt").write_text(report_text, encoding="utf-8")
    save_confusion_matrix_plot(labels, preds, artifacts["class_names"], run_dir / "re_eval_confusion_matrix.png")

    logger.info("重新评估完成 | loss=%.6f | acc=%.6f", test_loss, test_acc)
    logger.info("分类报告:\n%s", report_text)


if __name__ == "__main__":
    main()
