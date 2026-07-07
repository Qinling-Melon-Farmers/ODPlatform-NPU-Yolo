"""Factory helpers for frame sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from od_platform.frame_source.base import FrameSource
from od_platform.frame_source.camera import CameraSource
from od_platform.frame_source.config import CameraConfig
from od_platform.frame_source.image import ImageFolderSource, ImageSource
from od_platform.frame_source.types import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, SourceType
from od_platform.frame_source.video import VideoSource
from od_platform.frame_source.wrappers.aio import AsyncSource
from od_platform.frame_source.wrappers.threaded import BufferStrategy, ThreadedSource

STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://")


def detect_source_type(source: str | Path | int) -> SourceType:
    """Infer source type from camera id, URL scheme, suffix or directory."""
    source_text = str(source)
    if isinstance(source, int) or source_text.isdigit():
        return SourceType.CAMERA
    if source_text.lower().startswith(STREAM_SCHEMES):
        return SourceType.VIDEO

    path = Path(source)
    if path.is_dir():
        return SourceType.IMAGE_FOLDER

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return SourceType.IMAGE
    if suffix in VIDEO_EXTENSIONS:
        return SourceType.VIDEO
    raise ValueError(f"unsupported frame source: {source}")


def create_frame_source(
    source: str | Path | int,
    camera_config: CameraConfig | None = None,
    *,
    stride: int = 1,
    **_options: Any,
) -> FrameSource:
    """Create a concrete FrameSource and optionally set its stride."""
    if stride < 1:
        raise ValueError(f"stride must be greater than or equal to 1, got {stride}")
    inner = _build_source(source, camera_config)
    if stride > 1:
        inner.set_stride(stride)
    return inner


def create_threaded_source(
    source: str | Path | int,
    camera_config: CameraConfig | None = None,
    *,
    stride: int = 1,
    buffer: BufferStrategy = "latest",
    buffer_size: int = 1,
    warmup_frames: int = 0,
    read_timeout: float = 5.0,
    **options: Any,
) -> ThreadedSource:
    """Create a FrameSource wrapped by a background capture thread."""
    inner = create_frame_source(source, camera_config, stride=stride, **options)
    return ThreadedSource(
        inner,
        buffer=buffer,
        buffer_size=buffer_size,
        warmup_frames=warmup_frames,
        read_timeout=read_timeout,
    )


def create_async_source(
    source: str | Path | int,
    camera_config: CameraConfig | None = None,
    *,
    stride: int = 1,
    threaded: bool = True,
    buffer: BufferStrategy = "latest",
    buffer_size: int = 1,
    warmup_frames: int = 0,
    read_timeout: float = 5.0,
    **options: Any,
) -> AsyncSource:
    """Create an async wrapper around a frame source."""
    inner: FrameSource
    if threaded:
        inner = create_threaded_source(
            source,
            camera_config,
            stride=stride,
            buffer=buffer,
            buffer_size=buffer_size,
            warmup_frames=warmup_frames,
            read_timeout=read_timeout,
            **options,
        )
    else:
        inner = create_frame_source(source, camera_config, stride=stride, **options)
    return AsyncSource(inner)


def _build_source(source: str | Path | int, camera_config: CameraConfig | None) -> FrameSource:
    source_type = detect_source_type(source)
    if source_type == SourceType.CAMERA:
        camera_id = int(source)
        config = camera_config or CameraConfig()
        config = config.model_copy(update={"camera_id": camera_id})
        return CameraSource(config)
    if source_type == SourceType.VIDEO:
        source_text = str(source)
        if not source_text.lower().startswith(STREAM_SCHEMES) and not Path(source_text).exists():
            raise ValueError(f"video path does not exist: {source}")
        return VideoSource(source)
    if source_type == SourceType.IMAGE:
        path = Path(source)
        if not path.exists():
            raise ValueError(f"image path does not exist: {source}")
        return ImageSource(path)
    if source_type == SourceType.IMAGE_FOLDER:
        path = Path(source)
        if not path.exists():
            raise ValueError(f"image folder does not exist: {source}")
        return ImageFolderSource(path)
    raise ValueError(f"unsupported frame source: {source}")
