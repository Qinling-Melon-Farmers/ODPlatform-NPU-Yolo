"""YOLO validation runtime configuration."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator

from od_platform.common.constants import Task
from od_platform.runtime_config.base import BaseRuntimeConfig


class YOLOValConfig(BaseRuntimeConfig):
    """Pydantic model for one YOLO validation run."""

    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = BaseRuntimeConfig.FRAMEWORK_ONLY_FIELDS | {
        "task",
        "extra_args",
    }

    task: str = Field(default=Task.DETECT, description="Vision task type for run organization.")
    split: str = Field(default="val", description="Dataset split to validate: train, val or test.")
    conf: float | None = Field(default=0.001, ge=0.0, le=1.0, description="Confidence threshold for validation.")
    iou: float = Field(default=0.6, ge=0.0, le=1.0, description="NMS IoU threshold.")
    max_det: int = Field(default=300, ge=1, description="Maximum detections per image.")
    half: bool = Field(default=True, description="Use half precision when supported.")
    plots: bool = Field(default=True, description="Generate validation plots.")
    save_json: bool = Field(default=True, description="Save COCO-style JSON predictions.")
    save_hybrid: bool = Field(default=False, description="Save hybrid labels.")
    dnn: bool = Field(default=False, description="Use OpenCV DNN backend.")
    mask_ratio: int = Field(default=4, ge=1, description="Segmentation mask downsample ratio.")
    overlap_mask: bool = Field(default=True, description="Allow overlapped masks for segmentation.")
    extra_args: dict[str, Any] = Field(default_factory=dict, description="Additional Ultralytics kwargs.")

    @field_validator("task")
    @classmethod
    def _validate_task(cls, value: str) -> str:
        return Task.ensure_end_to_end(value)

    @field_validator("split")
    @classmethod
    def _validate_split(cls, value: str) -> str:
        valid = {"train", "val", "test"}
        if value not in valid:
            raise ValueError(f"unsupported split {value!r}, expected one of {sorted(valid)}")
        return value

    @model_validator(mode="after")
    def _validate_task_specific_fields(self) -> YOLOValConfig:
        if self.mask_ratio != 4 or not self.overlap_mask:
            raise ValueError("mask_ratio and overlap_mask are reserved for future segment support")
        return self

    def to_ultralytics_kwargs(self) -> dict[str, Any]:
        kwargs = super().to_ultralytics_kwargs()
        kwargs.update(self.extra_args)
        return kwargs
