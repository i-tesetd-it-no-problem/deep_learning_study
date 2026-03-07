from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from src.engine.evaluator import calculate_accuracy, evaluate_one_epoch
from src.utils.checkpoint import save_artifacts, save_checkpoint
from src.utils.plotting import save_confusion_matrix_plot, save_history_plot, save_lr_plot


@dataclass
class EarlyStopping:
    patience: int = 20
    min_delta: float = 1e-4
    best_loss: float = float("inf")
    counter: int = 0
    should_stop: bool = False

    def step(self, current_loss: float) -> None:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True



def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, gradient_clip):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if gradient_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        correct, batch_size = calculate_accuracy(logits, y)
        total_correct += correct
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc



def run_training(
    model,
    train_loader,
    val_loader,
    test_loader,
    data_bundle,
    config: Dict[str, Any],
    run_dir: str | Path,
    device: torch.device,
    logger,
) -> Tuple[Any, Dict[str, List[float]], Dict[str, Any]]:
    run_dir = Path(run_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["scheduler"]["factor"],
        patience=config["scheduler"]["patience"],
        min_lr=config["scheduler"]["min_lr"],
    )
    early_stopping = EarlyStopping(
        patience=config["early_stopping"]["patience"],
        min_delta=config["early_stopping"]["min_delta"],
    )

    use_amp = bool(config["train"].get("use_amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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

    for epoch in range(config["train"]["epochs"]):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            gradient_clip=config["train"].get("gradient_clip"),
        )

        val_loss, val_acc, _, _ = evaluate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                {
                    "model_state_dict": best_model_state,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "input_dim": model.input_dim,
                    "output_dim": model.output_dim,
                    "hidden_dims": model.hidden_dims,
                    "dropout": model.dropout_p,
                    "use_bn": model.use_bn,
                    "config": config,
                },
                run_dir / "best_model.pt",
            )

        early_stopping.step(val_loss)
        elapsed = time.time() - start_time
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f | LR: %.6f | Time: %.2fs",
            epoch + 1,
            config["train"]["epochs"],
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
            elapsed,
        )

        if early_stopping.should_stop:
            logger.info("Early Stopping 触发，训练停止于 epoch=%d", epoch + 1)
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_acc, preds, labels = evaluate_one_epoch(model, test_loader, criterion, device)
    report_text = classification_report(
        labels,
        preds,
        target_names=data_bundle.class_names,
        digits=config["eval"].get("digits", 4),
        zero_division=0,
    )

    logger.info("最佳验证损失: %.6f | 最佳 Epoch: %d", best_val_loss, best_epoch)
    logger.info("测试集损失: %.6f | 测试集准确率: %.6f", test_loss, test_acc)
    logger.info("分类报告:\n%s", report_text)

    (run_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    save_artifacts(
        {
            "scaler": data_bundle.scaler,
            "label_encoder": data_bundle.label_encoder,
            "feature_names": data_bundle.feature_names,
            "class_names": data_bundle.class_names,
        },
        run_dir / "artifacts.joblib",
    )

    if config["eval"].get("save_plots", True):
        save_history_plot(history, run_dir / "history.png")
        save_lr_plot(history, run_dir / "lr_curve.png")
        save_confusion_matrix_plot(labels, preds, data_bundle.class_names, run_dir / "confusion_matrix.png")

    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    return model, history, metrics
