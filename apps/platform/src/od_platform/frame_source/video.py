"""Video-file frame source."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from od_platform.frame_source.base import FrameSource
from od_platform.frame_source.types import Frame, FrameInfo, SourceType

logger = logging.getLogger(__name__)


class VideoSource(FrameSource):
    """OpenCV video source with seek and stride support."""

    def __init__(self, video_path: str | Path) -> None:
        super().__init__(str(video_path))
        self.path = Path(video_path)
        self._cap: cv2.VideoCapture | None = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._total_frames = 0
        self._filename = self.path.name

    def open(self) -> bool:
        self._frame_index = 0
        self._cap = cv2.VideoCapture(self.source_path)
        if not self._cap.isOpened():
            logger.error("Failed to open video: %s", self.source_path)
            return False

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        raw_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if raw_fps <= 0:
            logger.warning("Video %s FPS is unknown; fallback to 30 FPS for timestamps", self._filename)
            self._fps = 30.0
        else:
            self._fps = raw_fps

        logger.info(
            "Video opened: %s, size=%sx%s, fps=%.1f, frames=%s",
            self._filename,
            self._width,
            self._height,
            self._fps,
            self._total_frames,
        )
        return True

    def read(self) -> Frame | None:
        if self._cap is None:
            return None

        if self._stride > 1 and self._frame_index > 0:
            for _ in range(self._stride - 1):
                if not self._cap.grab():
                    return None
                self._frame_index += 1

        ret, image = self._cap.read()
        if not ret:
            return None

        info = FrameInfo(
            width=self._width,
            height=self._height,
            source_type=SourceType.VIDEO,
            source_path=self.source_path,
            frame_index=self._frame_index,
            total_frames=self._total_frames,
            timestamp=self._frame_index / self._fps if self._fps > 0 else 0.0,
            fps=self._fps,
            filename=self._filename,
        )
        self._frame_index += 1
        return Frame(image=image, info=info)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Video closed: %s", self._filename)

    def get_source_type(self) -> SourceType:
        return SourceType.VIDEO

    def seek(self, frame: int | None = None, time_sec: float | None = None) -> bool:
        if self._cap is None:
            logger.error("Video is not open; cannot seek")
            return False
        if frame is None and time_sec is None:
            logger.error("Video seek requires frame or time_sec")
            return False

        target = int(time_sec * self._fps) if time_sec is not None else int(frame)
        target = max(0, target)
        if self._total_frames > 0:
            target = min(target, self._total_frames - 1)

        ok = bool(self._cap.set(cv2.CAP_PROP_POS_FRAMES, target))
        if ok:
            self._frame_index = target
            logger.debug("Video seeked to frame %s", target)
        else:
            logger.warning("Video seek to frame %s failed", target)
        return ok

    def seekable(self) -> bool:
        return True

    @property
    def duration(self) -> float:
        if self._fps > 0 and self._total_frames > 0:
            return self._total_frames / self._fps
        return 0.0


VideoFrameSource = VideoSource
