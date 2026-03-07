from __future__ import annotations

from typing import List, Tuple

import torch



def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total



def evaluate_one_epoch(model, data_loader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

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
