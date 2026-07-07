"""Async wrapper for frame sources."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from od_platform.frame_source.base import FrameSource, FrameSourceError
from od_platform.frame_source.types import Frame


class AsyncSource:
    """Expose a blocking FrameSource through async context and iteration protocols."""

    def __init__(self, inner: FrameSource) -> None:
        self._inner = inner

    @property
    def inner(self) -> FrameSource:
        return self._inner

    async def open(self) -> bool:
        opened = await asyncio.to_thread(self._inner.open)
        if not opened:
            return False
        return True

    async def read(self) -> Frame | None:
        return await asyncio.to_thread(self._inner.read)

    async def close(self) -> None:
        await asyncio.to_thread(self._inner.close)

    async def __aenter__(self) -> AsyncSource:
        if not await self.open():
            raise FrameSourceError(f"failed to open frame source: {self._inner.source_path}")
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb) -> bool:
        await self.close()
        return False

    def __aiter__(self) -> AsyncIterator[Frame]:
        return self

    async def __anext__(self) -> Frame:
        frame = await self.read()
        if frame is None:
            raise StopAsyncIteration
        return frame
