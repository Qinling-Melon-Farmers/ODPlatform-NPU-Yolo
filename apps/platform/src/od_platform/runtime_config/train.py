"""YOLO training runtime configuration."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field, field_validator

from od_platform.common.constants import Task
from od_platform.runtime_config.base import BaseRuntimeConfig


class YOLOTrainConfig(BaseRuntimeConfig):
    """Pydantic model for one YOLO training run."""

    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = BaseRuntimeConfig.FRAMEWORK_ONLY_FIELDS | {
        "archive_weights",
        "copy_archive",
        "extra_args",
    }

    task: str = Field(default=Task.DETECT, description="Vision task type.")
    epochs: int = Field(default=100, ge=1, description="Training epochs.")
    patience: int = Field(default=100, ge=0, description="Early-stopping patience.")
    optimizer: str = Field(default="auto", description="Optimizer name.")
    lr0: float = Field(default=0.01, gt=0, description="Initial learning rate.")
    lrf: float = Field(default=0.01, gt=0, description="Final learning-rate factor.")
    pretrained: bool | str = Field(default=True, description="Pretrained weights flag or path.")
    resume: bool = Field(default=False, description="Resume training.")
    plots: bool = Field(default=True, description="Generate training plots.")
    archive_weights: bool = Field(default=True, description="Archive best.pt and last.pt after training.")
    copy_archive: bool = Field(default=True, description="Copy weights to archive instead of moving them.")
    extra_args: dict[str, Any] = Field(default_factory=dict, description="Additional Ultralytics kwargs.")

    @field_validator("task")
    @classmethod
    def _validate_task(cls, value: str) -> str:
        if value not in Task.all():
            raise ValueError(f"unsupported task {value!r}, expected one of {Task.all()}")
        return value

    def to_ultralytics_kwargs(self) -> dict[str, Any]:
        kwargs = super().to_ultralytics_kwargs()
        kwargs.update(self.extra_args)
        return kwargs
