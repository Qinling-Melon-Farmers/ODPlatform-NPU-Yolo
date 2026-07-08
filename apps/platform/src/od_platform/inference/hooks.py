"""Inference lifecycle hooks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameEvent:
    """Event emitted after one frame has been rendered."""

    frame_index: int
    image: np.ndarray
    annotated: np.ndarray
    detections: list[dict[str, Any]]


@dataclass
class ProgressEvent:
    """Low-frequency progress event emitted during long inference runs."""

    frame_index: int
    total_frames: int | None
    elapsed_seconds: float
    loop_fps: float
    detections_total: int


@dataclass
class InferHooks:
    """Optional callbacks for UI, web, or background job integrations."""

    on_frame: Callable[[FrameEvent], None] | None = None
    on_progress: Callable[[ProgressEvent], None] | None = None
    on_complete: Callable[[Any], None] | None = None
    on_error: Callable[[Exception], None] | None = None
    progress_interval_frames: int = 30

    def fire_frame(self, event: FrameEvent) -> None:
        if self.on_frame is not None:
            self._safe_call(self.on_frame, event, "on_frame")

    def fire_progress(self, event: ProgressEvent) -> None:
        if self.on_progress is not None:
            self._safe_call(self.on_progress, event, "on_progress")

    def fire_complete(self, result: Any) -> None:
        if self.on_complete is not None:
            self._safe_call(self.on_complete, result, "on_complete")

    def fire_error(self, exc: Exception) -> None:
        if self.on_error is not None:
            self._safe_call(self.on_error, exc, "on_error")

    @staticmethod
    def _safe_call(callback: Callable[[Any], None], value: Any, name: str) -> None:
        try:
            callback(value)
        except Exception as exc:
            logger.warning("%s callback failed and was ignored: %s", name, exc)
