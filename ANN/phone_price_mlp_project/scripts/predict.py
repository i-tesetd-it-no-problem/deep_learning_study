from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mlp import TabularMLP
from src.utils.checkpoint import load_artifacts, load_checkpoint



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with trained model")
    parser.add_argument("--run-dir", type=str, required=True, help="训练输出目录")
    parser.add_argument("--input-json", type=str, default=None, help="单样本 JSON 字符串")
    parser.add_argument("--input-csv", type=str, default=None, help="批量样本 CSV 路径")
    parser.add_argument("--save-path", type=str, default=None, help="批量预测结果保存路径")
    return parser.parse_args()



def _build_input_df(args, feature_names):
    if args.input_json:
        sample = json.loads(args.input_json)
        return pd.DataFrame([sample], columns=feature_names)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        missing_cols = [col for col in feature_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"输入 CSV 缺少这些特征列: {missing_cols}")
        return df[feature_names].copy()

    raise ValueError("必须至少提供 --input-json 或 --input-csv 其中一个")



def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(run_dir / "best_model.pt", map_location=device)
    artifacts = load_artifacts(run_dir / "artifacts.joblib")

    feature_names = artifacts["feature_names"]
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]

    input_df = _build_input_df(args, feature_names)
    if input_df.isnull().any().any():
        raise ValueError("输入数据存在缺失值，请先处理后再推理")

    x = input_df.astype("float32").values
    x = scaler.transform(x)
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    model = TabularMLP(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
        use_bn=checkpoint["use_bn"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_ids = torch.argmax(probs, dim=1).cpu().numpy()
        pred_labels = label_encoder.inverse_transform(pred_ids)
        pred_conf = probs.max(dim=1).values.cpu().numpy()

    result_df = input_df.copy()
    result_df["pred_id"] = pred_ids
    result_df["pred_label"] = pred_labels
    result_df["confidence"] = pred_conf

    if args.input_json:
        print(result_df.to_json(force_ascii=False, orient="records", indent=2))
    else:
        save_path = Path(args.save_path or (run_dir / "predictions.csv"))
        result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"预测完成，结果已保存到: {save_path}")


if __name__ == "__main__":
    main()
