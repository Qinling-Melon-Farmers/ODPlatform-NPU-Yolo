"""Cancellation primitives for inference pipelines."""

from __future__ import annotations

import threading


class CancelToken:
    """Thread-safe cancellation signal checked by inference loops."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return whether cancellation was requested."""
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until cancelled or until timeout expires."""
        return self._event.wait(timeout)


class InferenceCancelled(Exception):
    """Raised by callers that want to distinguish user cancellation."""
