"""Backward-compatible OpenCV frame source entry points."""

from __future__ import annotations

from od_platform.frame_source.camera import CameraFrameSource, CameraSource
from od_platform.frame_source.factory import create_frame_source, detect_source_type
from od_platform.frame_source.image import (
    ImageFolderFrameSource,
    ImageFolderSource,
    ImageFrameSource,
    ImageSource,
)
from od_platform.frame_source.video import VideoFrameSource, VideoSource

__all__ = [
    "CameraFrameSource",
    "CameraSource",
    "ImageFolderFrameSource",
    "ImageFolderSource",
    "ImageFrameSource",
    "ImageSource",
    "VideoFrameSource",
    "VideoSource",
    "create_frame_source",
    "detect_source_type",
]
