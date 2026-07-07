"""Wrappers that extend any FrameSource implementation."""

from __future__ import annotations

from od_platform.frame_source.wrappers.aio import AsyncSource
from od_platform.frame_source.wrappers.threaded import BufferStrategy, ThreadedSource

__all__ = ["AsyncSource", "BufferStrategy", "ThreadedSource"]
