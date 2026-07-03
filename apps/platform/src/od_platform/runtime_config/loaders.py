"""Load runtime configuration from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from od_platform.common import paths
from od_platform.runtime_config.train import YOLOTrainConfig


def resolve_runtime_config(ref: str | Path) -> Path:
    """Resolve a runtime config name or path."""
    path = Path(ref)
    if path.is_absolute() or len(path.parts) > 1:
        return path.resolve()
    name = path.name if path.name.endswith((".yaml", ".yml")) else f"{path.name}.yaml"
    return (paths.RUNTIME_CONFIGS_DIR / name).resolve()


def load_yaml_dict(path: Path) -> dict[str, Any]:
    """Read a YAML file and require a mapping at the top level."""
    if not path.exists():
        raise FileNotFoundError(f"runtime config does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"runtime config top-level must be a mapping: {path}")
    return data


def load_train_config(ref: str | Path) -> YOLOTrainConfig:
    """Load and validate one YOLO training config."""
    return YOLOTrainConfig.model_validate(load_yaml_dict(resolve_runtime_config(ref)))
