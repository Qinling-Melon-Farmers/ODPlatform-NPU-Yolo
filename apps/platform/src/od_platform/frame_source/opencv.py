"""OpenCV-backed frame source implementations."""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2

from od_platform.frame_source.base import FrameSource, FrameSourceError
from od_platform.frame_source.config import CameraConfig
from od_platform.frame_source.types import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    Frame,
    FrameInfo,
    SourceType,
)


class ImageFrameSource(FrameSource):
    """Single image source."""

    def __init__(self, source_path: str | Path) -> None:
        super().__init__(str(source_path))
        self.path = Path(source_path)
        self._opened = False
        self._consumed = False

    def open(self) -> bool:
        self._opened = self.path.is_file()
        self._consumed = False
        return self._opened

    def read(self) -> Frame | None:
        if not self._opened or self._consumed:
            return None
        image = cv2.imread(str(self.path))
        if image is None:
            raise FrameSourceError(f"failed to read image: {self.path}")
        self._consumed = True
        height, width = image.shape[:2]
        return Frame(
            image=image,
            info=FrameInfo(
                width=width,
                height=height,
                source_type=SourceType.IMAGE,
                source_path=str(self.path),
                frame_index=0,
                total_frames=1,
                filename=self.path.name,
            ),
        )

    def close(self) -> None:
        self._opened = False

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE

    def seek(self, frame: int | None = None, time_sec: float | None = None) -> bool:
        if frame in (None, 0) and time_sec in (None, 0.0):
            self._consumed = False
            return True
        return False

    def seekable(self) -> bool:
        return True


class ImageFolderFrameSource(FrameSource):
    """Read images from a folder in lexical order."""

    def __init__(self, source_path: str | Path) -> None:
        super().__init__(str(source_path))
        self.path = Path(source_path)
        self._images: list[Path] = []
        self._cursor = 0

    def open(self) -> bool:
        if not self.path.is_dir():
            return False
        self._images = sorted(
            item for item in self.path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        self._cursor = 0
        self._frame_index = 0
        return True

    def read(self) -> Frame | None:
        while self._cursor < len(self._images):
            path = self._images[self._cursor]
            index = self._cursor
            self._cursor += self._stride
            image = cv2.imread(str(path))
            if image is None:
                continue
            height, width = image.shape[:2]
            self._frame_index = index + 1
            return Frame(
                image=image,
                info=FrameInfo(
                    width=width,
                    height=height,
                    source_type=SourceType.IMAGE_FOLDER,
                    source_path=str(self.path),
                    frame_index=index,
                    total_frames=len(self._images),
                    filename=path.name,
                ),
            )
        return None

    def close(self) -> None:
        self._images = []
        self._cursor = 0

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE_FOLDER

    def seek(self, frame: int | None = None, time_sec: float | None = None) -> bool:
        if frame is None:
            frame = 0
        if frame < 0 or frame >= len(self._images):
            return False
        self._cursor = frame
        return True

    def seekable(self) -> bool:
        return True


class VideoFrameSource(FrameSource):
    """OpenCV video file source."""

    def __init__(self, source_path: str | Path) -> None:
        super().__init__(str(source_path))
        self.path = Path(source_path)
        self._cap = None
        self._total_frames: int | None = None
        self._fps: float | None = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            return False
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._total_frames = total if total > 0 else None
        self._fps = fps if fps > 0 else None
        self._frame_index = 0
        return True

    def read(self) -> Frame | None:
        if self._cap is None:
            return None
        ret, image = self._cap.read()
        if not ret:
            return None
        raw_index = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if self._stride > 1:
            next_index = raw_index + self._stride
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, next_index)
        height, width = image.shape[:2]
        timestamp = float(self._cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        self._frame_index = raw_index + 1
        return Frame(
            image=image,
            info=FrameInfo(
                width=width,
                height=height,
                source_type=SourceType.VIDEO,
                source_path=str(self.path),
                frame_index=raw_index,
                total_frames=self._total_frames,
                timestamp=timestamp,
                fps=self._fps,
                filename=self.path.name,
            ),
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_source_type(self) -> SourceType:
        return SourceType.VIDEO

    def seek(self, frame: int | None = None, time_sec: float | None = None) -> bool:
        if self._cap is None:
            return False
        if frame is not None:
            return bool(self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame))
        if time_sec is not None:
            return bool(self._cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0))
        return False

    def seekable(self) -> bool:
        return True


class CameraFrameSource(FrameSource):
    """OpenCV camera source with backend, codec and FPS control."""

    _BACKENDS = {
        "auto": 0,
        "msmf": cv2.CAP_MSMF,
        "dshow": cv2.CAP_DSHOW,
        "v4l2": cv2.CAP_V4L2,
    }

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        super().__init__(self.config.camera_id)
        self._cap = None
        self._actual_width = 0
        self._actual_height = 0
        self._actual_fps: float | None = None

    def open(self) -> bool:
        if self.config.backend == "msmf":
            os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
        backend = self._BACKENDS[self.config.backend]
        self._cap = cv2.VideoCapture(self.config.camera_id, backend) if backend else cv2.VideoCapture(self.config.camera_id)
        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.codec))
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        for _ in range(self.config.warmup_reads):
            self._cap.read()

        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._actual_fps = fps if fps > 0 else None
        self._frame_index = 0
        self._start_time = time.time()
        return True

    def read(self) -> Frame | None:
        if self._cap is None:
            return None
        ret, image = self._cap.read()
        if not ret:
            return None
        height, width = image.shape[:2]
        index = self._frame_index
        self._frame_index += 1
        return Frame(
            image=image,
            info=FrameInfo(
                width=width,
                height=height,
                source_type=SourceType.CAMERA,
                source_path=str(self.config.camera_id),
                frame_index=index,
                timestamp=time.time() - self._start_time,
                fps=self._actual_fps,
                metadata={
                    "backend": self.config.backend,
                    "codec": self.config.codec,
                    "requested_width": self.config.width,
                    "requested_height": self.config.height,
                    "requested_fps": self.config.fps,
                    "actual_width": self._actual_width,
                    "actual_height": self._actual_height,
                },
            ),
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_source_type(self) -> SourceType:
        return SourceType.CAMERA

    def metadata(self) -> dict[str, object]:
        payload = super().metadata()
        payload.update(
            {
                "camera_id": self.config.camera_id,
                "backend": self.config.backend,
                "codec": self.config.codec,
                "requested_resolution": self.config.get_resolution(),
                "actual_width": self._actual_width,
                "actual_height": self._actual_height,
                "actual_fps": self._actual_fps,
            }
        )
        return payload


def detect_source_type(source: str | Path | int) -> SourceType:
    """Infer source type from camera id, suffix or directory."""
    if isinstance(source, int) or str(source).isdigit():
        return SourceType.CAMERA
    path = Path(source)
    if path.is_dir():
        return SourceType.IMAGE_FOLDER
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return SourceType.IMAGE
    if suffix in VIDEO_EXTENSIONS:
        return SourceType.VIDEO
    raise ValueError(f"unsupported frame source: {source}")


def create_frame_source(source: str | Path | int, *, camera_config: CameraConfig | None = None) -> FrameSource:
    """Create an OpenCV-backed frame source for the given input."""
    source_type = detect_source_type(source)
    if source_type == SourceType.CAMERA:
        config = camera_config or CameraConfig(camera_id=int(source))
        return CameraFrameSource(config)
    if source_type == SourceType.IMAGE:
        return ImageFrameSource(source)
    if source_type == SourceType.IMAGE_FOLDER:
        return ImageFolderFrameSource(source)
    if source_type == SourceType.VIDEO:
        return VideoFrameSource(source)
    raise ValueError(f"unsupported frame source: {source}")
