"""Public builders for runtime configs."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.runtime_config.loaders import CLILoader, YAMLLoader
from od_platform.runtime_config.merger import ConfigMerger, ConfigSource
from od_platform.runtime_config.train import YOLOTrainConfig
from od_platform.runtime_config.val import YOLOValConfig

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def _build_config(
    config_class: type[ConfigT],
    yaml_path: str | Path | None,
    cli_args: Namespace | None,
    *,
    dry_run: bool,
) -> tuple[ConfigT | None, ConfigMerger]:
    merger = ConfigMerger()
    sources: list[tuple[ConfigSource, dict]] = []
    if yaml_path is not None:
        sources.append((ConfigSource.YAML, YAMLLoader().load(yaml_path)))
    if cli_args is not None:
        sources.append((ConfigSource.CLI, CLILoader().load(cli_args)))
    if dry_run:
        merger.preview(config_class, sources=sources)
        return None, merger
    return merger.merge(config_class, sources=sources), merger


def build_train_config(
    yaml_path: str | Path | None = "train.yaml",
    cli_args: Namespace | None = None,
    *,
    dry_run: bool = False,
) -> tuple[YOLOTrainConfig | None, ConfigMerger]:
    """Build a training config from defaults, YAML and optional CLI args."""
    return _build_config(YOLOTrainConfig, yaml_path, cli_args, dry_run=dry_run)


def build_val_config(
    yaml_path: str | Path | None = "val.yaml",
    cli_args: Namespace | None = None,
    *,
    dry_run: bool = False,
) -> tuple[YOLOValConfig | None, ConfigMerger]:
    """Build a validation config from defaults, YAML and optional CLI args."""
    return _build_config(YOLOValConfig, yaml_path, cli_args, dry_run=dry_run)


def build_infer_config(
    yaml_path: str | Path | None = "infer.yaml",
    cli_args: Namespace | None = None,
    *,
    dry_run: bool = False,
) -> tuple[YOLOInferConfig | None, ConfigMerger]:
    """Build an inference config from defaults, YAML and optional CLI args."""
    return _build_config(YOLOInferConfig, yaml_path, cli_args, dry_run=dry_run)
