"""Frame source abstractions for realtime and offline inputs."""

from od_platform.frame_source.base import FrameSource, FrameSourceError
from od_platform.frame_source.camera import CameraSource
from od_platform.frame_source.config import CameraBackend, CameraCodec, CameraConfig
from od_platform.frame_source.factory import (
    create_async_source,
    create_frame_source,
    create_threaded_source,
    detect_source_type,
)
from od_platform.frame_source.image import ImageFolderSource, ImageSource
from od_platform.frame_source.opencv import (
    CameraFrameSource,
    ImageFolderFrameSource,
    ImageFrameSource,
    VideoFrameSource,
)
from od_platform.frame_source.types import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    Frame,
    FrameInfo,
    SourceType,
)
from od_platform.frame_source.video import VideoSource
from od_platform.frame_source.wrappers import AsyncSource, BufferStrategy, ThreadedSource

__version__ = "0.1.0"

__all__ = [
    "AsyncSource",
    "BufferStrategy",
    "CameraBackend",
    "CameraCodec",
    "CameraConfig",
    "CameraFrameSource",
    "CameraSource",
    "Frame",
    "FrameInfo",
    "FrameSource",
    "FrameSourceError",
    "IMAGE_EXTENSIONS",
    "ImageFolderFrameSource",
    "ImageFolderSource",
    "ImageFrameSource",
    "ImageSource",
    "SourceType",
    "ThreadedSource",
    "VIDEO_EXTENSIONS",
    "VideoFrameSource",
    "VideoSource",
    "create_async_source",
    "create_frame_source",
    "create_threaded_source",
    "detect_source_type",
    "__version__",
]
