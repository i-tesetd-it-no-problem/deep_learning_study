from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import torch



def save_checkpoint(checkpoint: Dict[str, Any], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)



def load_checkpoint(load_path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {load_path}")
    return torch.load(load_path, map_location=map_location)



def save_artifacts(artifacts: Dict[str, Any], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, save_path)



def load_artifacts(load_path: str | Path) -> Dict[str, Any]:
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"工件文件不存在: {load_path}")
    return joblib.load(load_path)
