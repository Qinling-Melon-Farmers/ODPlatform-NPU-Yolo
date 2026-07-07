"""Concrete frame source implementations."""

from __future__ import annotations

from od_platform.frame_source.sources.camera import CameraSource
from od_platform.frame_source.sources.image import ImageFolderSource, ImageSource
from od_platform.frame_source.sources.video import VideoSource

__all__ = [
    "CameraSource",
    "ImageFolderSource",
    "ImageSource",
    "VideoSource",
]
