"""Frame source abstractions for realtime and offline inputs."""

from od_platform.frame_source.base import FrameSource, FrameSourceError
from od_platform.frame_source.config import CameraBackend, CameraCodec, CameraConfig
from od_platform.frame_source.opencv import (
    CameraFrameSource,
    ImageFolderFrameSource,
    ImageFrameSource,
    VideoFrameSource,
    create_frame_source,
    detect_source_type,
)
from od_platform.frame_source.types import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    Frame,
    FrameInfo,
    SourceType,
)

__all__ = [
    "CameraBackend",
    "CameraCodec",
    "CameraConfig",
    "CameraFrameSource",
    "Frame",
    "FrameInfo",
    "FrameSource",
    "FrameSourceError",
    "IMAGE_EXTENSIONS",
    "ImageFolderFrameSource",
    "ImageFrameSource",
    "SourceType",
    "VIDEO_EXTENSIONS",
    "VideoFrameSource",
    "create_frame_source",
    "detect_source_type",
]
