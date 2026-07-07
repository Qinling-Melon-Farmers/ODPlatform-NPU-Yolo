"""Core data types for frame sources."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class SourceType(str, Enum):
    """Input source type."""

    CAMERA = "camera"
    IMAGE = "image"
    VIDEO = "video"
    IMAGE_FOLDER = "image_folder"


IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"})


@dataclass(frozen=True)
class FrameInfo:
    """Metadata attached to one frame."""

    width: int
    height: int
    source_type: SourceType
    source_path: str
    frame_index: int = 0
    total_frames: int | None = None
    timestamp: float = 0.0
    fps: float | None = None
    filename: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass
class Frame:
    """Image array and metadata read from a source."""

    image: np.ndarray
    info: FrameInfo

    @property
    def resolution(self) -> tuple[int, int]:
        return self.info.resolution

    @property
    def width(self) -> int:
        return self.info.width

    @property
    def height(self) -> int:
        return self.info.height
