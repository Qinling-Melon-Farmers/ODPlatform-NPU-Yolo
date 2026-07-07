"""Load runtime configuration from YAML files."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

from od_platform.common import paths
from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.runtime_config.train import YOLOTrainConfig
from od_platform.runtime_config.val import YOLOValConfig


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    """Drop only None values; keep False, 0 and empty strings as explicit inputs."""
    return {key: value for key, value in data.items() if value is not None}


class YAMLLoader:
    """Load YAML runtime config files into dictionaries."""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else paths.RUNTIME_CONFIGS_DIR

    def load(self, ref: str | Path) -> dict[str, Any]:
        return load_yaml_dict(self._resolve_path(ref))

    def _resolve_path(self, ref: str | Path) -> Path:
        path = Path(ref)
        if path.is_absolute() or len(path.parts) > 1:
            return path.resolve()
        name = path.name if path.name.endswith((".yaml", ".yml")) else f"{path.name}.yaml"
        return (self.config_dir / name).resolve()


class CLILoader:
    """Extract runtime config fields from an argparse Namespace."""

    CONTROL_FIELDS = {"config", "command", "dry_run", "executor", "func"}

    def load(self, args: Namespace, *, exclude: set[str] | None = None) -> dict[str, Any]:
        excluded = self.CONTROL_FIELDS | (exclude or set())
        return _drop_none({key: value for key, value in vars(args).items() if key not in excluded})


def resolve_runtime_config(ref: str | Path) -> Path:
    """Resolve a runtime config name or path."""
    return YAMLLoader()._resolve_path(ref)


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


def load_val_config(ref: str | Path) -> YOLOValConfig:
    """Load and validate one YOLO validation config."""
    return YOLOValConfig.model_validate(load_yaml_dict(resolve_runtime_config(ref)))


def load_infer_config(ref: str | Path) -> YOLOInferConfig:
    """Load and validate one YOLO inference config."""
    return YOLOInferConfig.model_validate(load_yaml_dict(resolve_runtime_config(ref)))
