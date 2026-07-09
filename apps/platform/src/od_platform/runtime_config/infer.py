"""YOLO inference runtime configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator

from od_platform.common.constants import Task
from od_platform.runtime_config.base import BaseRuntimeConfig


class YOLOInferConfig(BaseRuntimeConfig):
    """Pydantic model for one YOLO inference run."""

    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = BaseRuntimeConfig.FRAMEWORK_ONLY_FIELDS | {
        "data",
        "task",
        "extra_args",
    }

    task: str = Field(default=Task.DETECT, description="Vision task type for run organization.")
    source: str | Path | int | None = Field(default=None, description="Image, video, directory, URL or camera index.")
    conf: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold.")
    iou: float = Field(default=0.7, ge=0.0, le=1.0, description="NMS IoU threshold.")
    max_det: int = Field(default=300, ge=1, description="Maximum detections per image.")
    classes: list[int] | None = Field(default=None, description="Optional class id filter.")
    agnostic_nms: bool = Field(default=False, description="Class-agnostic NMS.")
    augment: bool = Field(default=False, description="Test-time augmentation.")
    vid_stride: int = Field(default=1, ge=1, description="Video frame stride.")
    stream: bool = Field(default=False, description="Return streaming generator.")
    stream_buffer: bool = Field(default=False, description="Buffer stream frames.")
    save_txt: bool = Field(default=False, description="Save prediction labels as YOLO txt.")
    save_conf: bool = Field(default=False, description="Save confidence values to txt labels.")
    save_crop: bool = Field(default=False, description="Save cropped detections.")
    save_frames: bool = Field(default=False, description="Save video frames.")
    show: bool = Field(default=False, description="Display prediction windows.")
    show_labels: bool = Field(default=True, description="Render labels.")
    show_conf: bool = Field(default=True, description="Render confidence scores.")
    show_boxes: bool = Field(default=True, description="Render boxes.")
    line_width: int | None = Field(default=None, ge=1, description="Rendered box line width.")
    retina_masks: bool = Field(default=False, description="High-resolution masks for segmentation.")
    visualize: bool = Field(default=False, description="Save feature visualizations.")
    embed: list[int] | None = Field(default=None, description="Layers used for embedding extraction.")
    extra_args: dict[str, Any] = Field(default_factory=dict, description="Additional Ultralytics kwargs.")

    @field_validator("task")
    @classmethod
    def _validate_task(cls, value: str) -> str:
        return Task.ensure_end_to_end(value)

    @field_validator("source")
    @classmethod
    def _normalize_source(cls, value: str | Path | int | None) -> str | None:
        if value is None:
            return None
        return str(value)

    @model_validator(mode="after")
    def _validate_output_flags(self) -> YOLOInferConfig:
        if self.save_conf and not self.save_txt:
            raise ValueError("save_conf=True requires save_txt=True")
        if self.stream_buffer and not self.stream:
            raise ValueError("stream_buffer=True requires stream=True")
        if self.retina_masks:
            raise ValueError("retina_masks=True is reserved for future segment support")
        return self

    def to_ultralytics_kwargs(self) -> dict[str, Any]:
        kwargs = super().to_ultralytics_kwargs()
        kwargs.update(self.extra_args)
        return kwargs
