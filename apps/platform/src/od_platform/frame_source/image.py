"""Image and image-folder frame sources."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from od_platform.frame_source.base import FrameSource
from od_platform.frame_source.types import IMAGE_EXTENSIONS, Frame, FrameInfo, SourceType

logger = logging.getLogger(__name__)


class ImageSource(FrameSource):
    """Single image source."""

    def __init__(self, image_path: str | Path) -> None:
        super().__init__(str(image_path))
        self.path = Path(image_path)
        self._image: np.ndarray | None = None
        self._read_count = 0
        self._file_name = self.path.name

    def open(self) -> bool:
        self._read_count = 0
        self._image = cv2.imread(self.source_path)
        if self._image is None:
            logger.error("Failed to read image: %s", self.source_path)
            return False
        height, width = self._image.shape[:2]
        logger.info("Image loaded: %s, shape: (%s x %s)", self.source_path, width, height)
        return True

    def read(self) -> Frame | None:
        if self._image is None or self._read_count > 0:
            return None
        height, width = self._image.shape[:2]
        info = FrameInfo(
            width=width,
            height=height,
            source_type=SourceType.IMAGE,
            source_path=str(self.source_path),
            frame_index=0,
            total_frames=1,
            filename=self._file_name,
        )
        self._read_count += 1
        return Frame(image=self._image.copy(), info=info)

    def close(self) -> None:
        self._image = None

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE

    def seek(self, frame: int | None = None, time_sec: float | None = None) -> bool:
        if frame in (None, 0) and time_sec in (None, 0.0):
            self._read_count = 0
            return True
        return False

    def seekable(self) -> bool:
        return True


class ImageFolderSource(FrameSource):
    """Read images from a folder in lexical order."""

    def __init__(self, folder_path: str | Path) -> None:
        super().__init__(str(folder_path))
        self.path = Path(folder_path)
        self._images: list[Path] = []
        self._cursor = 0

    def open(self) -> bool:
        if not self.path.is_dir():
            logger.error("Image folder not found: %s", self.path)
            return False
        self._images = sorted(
            item for item in self.path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        self._cursor = 0
        self._frame_index = 0
        logger.info("Image folder loaded: %s, images=%s", self.path, len(self._images))
        return True

    def read(self) -> Frame | None:
        while self._cursor < len(self._images):
            path = self._images[self._cursor]
            index = self._cursor
            self._cursor += self._stride
            image = cv2.imread(str(path))
            if image is None:
                logger.warning("Skip unreadable image: %s", path)
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


ImageFrameSource = ImageSource
ImageFolderFrameSource = ImageFolderSource
