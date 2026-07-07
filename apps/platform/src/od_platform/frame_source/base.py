"""Abstract protocol shared by all frame sources."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from od_platform.frame_source.types import Frame, SourceType

logger = logging.getLogger(__name__)


class FrameSourceError(RuntimeError):
    """Raised when a frame source cannot be opened or read safely."""


class FrameSource(ABC):
    """Unified interface for image, folder, video and camera inputs."""

    def __init__(self, source_path: str | int) -> None:
        self.source_path = source_path
        self._frame_index = 0
        self._stride = 1
        self._start_time = time.time()

    @abstractmethod
    def open(self) -> bool:
        """Open the input source and return whether it succeeded."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Read the next frame, or return None at end-of-stream."""

    @abstractmethod
    def close(self) -> None:
        """Close the input source and release resources."""

    @abstractmethod
    def get_source_type(self) -> SourceType:
        """Return the concrete source type."""

    def seek(self, frame: int | None = None, time_sec: float | None = None) -> bool:
        logger.warning("%s does not support seek", self.__class__.__name__)
        return False

    def seekable(self) -> bool:
        return False

    def set_stride(self, stride: int) -> None:
        """Set sampling stride. stride=2 means read every other frame."""
        if stride < 1:
            raise ValueError("stride must be greater than or equal to 1")
        if stride > 1:
            logger.debug("%s stride set to %s", self.__class__.__name__, stride)
        self._stride = stride

    def stride(self) -> int:
        return self._stride

    def metadata(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_type": self.get_source_type().value,
            "stride": self._stride,
        }

    def __enter__(self) -> FrameSource:
        if not self.open():
            raise FrameSourceError(f"failed to open frame source: {self.source_path}")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        self.close()
        return False

    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame
