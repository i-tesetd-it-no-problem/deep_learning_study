from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


ConfigDict = Dict[str, Any]


def load_yaml_config(config_path: str | Path) -> ConfigDict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("配置文件内容必须是字典结构")

    return config
