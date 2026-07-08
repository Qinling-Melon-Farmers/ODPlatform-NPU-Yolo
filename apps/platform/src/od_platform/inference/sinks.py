"""Output sinks for rendered inference frames."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from od_platform.frame_source import SourceType

logger = logging.getLogger(__name__)


class OutputSink(ABC):
    """Abstract destination for annotated frames."""

    @abstractmethod
    def open(self, output_dir: Path, source_type: SourceType) -> None:
        """Initialize the sink."""

    @abstractmethod
    def write(self, frame, annotated: np.ndarray) -> None:
        """Write one annotated frame."""

    @abstractmethod
    def close(self) -> None:
        """Release sink resources."""


class NullSink(OutputSink):
    """Sink that discards all frames."""

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        return None

    def write(self, frame, annotated: np.ndarray) -> None:
        return None

    def close(self) -> None:
        return None


class LocalFileSink(OutputSink):
    """Write rendered frames to jpg files or mp4 video."""

    def __init__(self) -> None:
        self.output_dir: Path | None = None
        self._is_stream = False
        self._video = None

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self._is_stream = source_type in {SourceType.CAMERA, SourceType.VIDEO}

    def write(self, frame, annotated: np.ndarray) -> None:
        import cv2

        if self.output_dir is None:
            return
        try:
            if self._is_stream:
                if self._video is None:
                    height, width = annotated.shape[:2]
                    fps = float(frame.info.fps or 30.0)
                    target = self.output_dir / "output.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._video = cv2.VideoWriter(str(target), fourcc, fps, (width, height))
                self._video.write(annotated)
                return

            filename = frame.info.filename or f"frame_{frame.info.frame_index:06d}.jpg"
            target = self.output_dir / f"{Path(filename).stem}.jpg"
            cv2.imwrite(str(target), annotated)
        except Exception as exc:
            logger.warning("failed to write inference frame, skipped: %s", exc)

    def close(self) -> None:
        if self._video is not None:
            try:
                self._video.release()
            finally:
                self._video = None
