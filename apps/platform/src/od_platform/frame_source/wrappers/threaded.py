"""Threaded frame source wrapper."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Literal

from od_platform.frame_source.base import FrameSource
from od_platform.frame_source.types import Frame, SourceType

logger = logging.getLogger(__name__)

BufferStrategy = Literal["latest", "bounded"]

_EOS = object()


class ThreadedSource(FrameSource):
    """Read frames in a background thread and expose the same FrameSource protocol."""

    def __init__(
        self,
        inner: FrameSource,
        *,
        buffer: BufferStrategy = "latest",
        buffer_size: int = 1,
        warmup_frames: int = 0,
        read_timeout: float = 5.0,
    ) -> None:
        super().__init__(inner.source_path)
        if buffer not in ("latest", "bounded"):
            raise ValueError("buffer must be 'latest' or 'bounded'")
        if buffer_size < 1:
            raise ValueError("buffer_size must be greater than or equal to 1")
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be greater than or equal to 0")
        self._inner = inner
        self._buffer = buffer
        self._read_timeout = read_timeout
        self._warmup_frames = warmup_frames
        self._queue: queue.Queue[Frame | object] = queue.Queue(maxsize=1 if buffer == "latest" else buffer_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._opened = False

    @property
    def inner(self) -> FrameSource:
        return self._inner

    def open(self) -> bool:
        self.close()
        self._stop.clear()
        self._drain_queue()
        if not self._inner.open():
            return False
        self._opened = True
        self._thread = threading.Thread(target=self._worker, name=f"{self.__class__.__name__}-{id(self)}", daemon=True)
        self._thread.start()
        return True

    def read(self) -> Frame | None:
        if not self._opened:
            return None
        try:
            item = self._queue.get(timeout=self._read_timeout)
        except queue.Empty:
            return None
        if item is _EOS:
            self._put_eos()
            return None
        return item

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(self._read_timeout, 0.1))
        self._thread = None
        if self._opened:
            self._inner.close()
        self._opened = False
        self._drain_queue()

    def get_source_type(self) -> SourceType:
        return self._inner.get_source_type()

    def seek(self, frame: int | None = None, time_sec: float | None = None) -> bool:
        logger.warning("ThreadedSource does not support seek while running; seek the inner source before wrapping")
        return False

    def seekable(self) -> bool:
        return False

    def set_stride(self, stride: int) -> None:
        self._inner.set_stride(stride)
        self._stride = self._inner.stride()

    def stride(self) -> int:
        return self._inner.stride()

    def metadata(self) -> dict[str, object]:
        payload = super().metadata()
        payload.update(
            {
                "buffer": self._buffer,
                "inner": self._inner.metadata(),
                "thread_alive": self._thread.is_alive() if self._thread is not None else False,
            }
        )
        return payload

    def _worker(self) -> None:
        try:
            for _ in range(self._warmup_frames):
                if self._stop.is_set() or self._inner.read() is None:
                    break
            while not self._stop.is_set():
                frame = self._inner.read()
                if frame is None:
                    break
                self._put_frame(frame)
        except Exception:
            logger.exception("ThreadedSource worker failed")
        finally:
            self._put_eos()

    def _put_frame(self, frame: Frame) -> None:
        if self._buffer == "latest":
            self._drain_queue()
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass
            return

        while not self._stop.is_set():
            try:
                self._queue.put(frame, timeout=0.1)
                return
            except queue.Full:
                continue

    def _put_eos(self) -> None:
        if self._buffer == "latest":
            self._drain_queue()
        try:
            self._queue.put_nowait(_EOS)
        except queue.Full:
            pass

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
