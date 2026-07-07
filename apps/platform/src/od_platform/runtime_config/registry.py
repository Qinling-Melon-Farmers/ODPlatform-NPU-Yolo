"""Registry of runtime config models."""

from __future__ import annotations

from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.runtime_config.train import YOLOTrainConfig
from od_platform.runtime_config.val import YOLOValConfig

CONFIG_REGISTRY: dict[str, tuple[type, str]] = {
    "train": (YOLOTrainConfig, "YOLO training config"),
    "val": (YOLOValConfig, "YOLO validation config"),
    "infer": (YOLOInferConfig, "YOLO inference config"),
}


def get_config_class(name: str) -> type:
    """Return a registered runtime config model class."""
    if name not in CONFIG_REGISTRY:
        raise ValueError(f"unknown runtime config {name!r}, expected one of {sorted(CONFIG_REGISTRY)}")
    return CONFIG_REGISTRY[name][0]


def list_config_names() -> list[str]:
    """Return registered runtime config names."""
    return list(CONFIG_REGISTRY)
