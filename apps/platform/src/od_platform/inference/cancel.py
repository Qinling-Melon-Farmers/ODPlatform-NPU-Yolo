"""Cancellation primitives for inference pipelines."""

from __future__ import annotations

import threading
import time


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


class PauseToken:
    """Thread-safe pause/resume signal for long-running inference loops."""

    def __init__(self) -> None:
        self._paused = threading.Event()

    def pause(self) -> None:
        """Request pipeline pause."""
        self._paused.set()

    def resume(self) -> None:
        """Resume a paused pipeline."""
        self._paused.clear()

    def toggle(self) -> bool:
        """Toggle pause state and return the new paused state."""
        if self.is_paused():
            self.resume()
            return False
        self.pause()
        return True

    def is_paused(self) -> bool:
        """Return whether pause was requested."""
        return self._paused.is_set()

    def wait_while_paused(self, cancel_token: CancelToken | None = None, *, interval: float = 0.05) -> bool:
        """Block while paused.

        Returns False when cancellation was requested while waiting.
        """
        while self.is_paused():
            if cancel_token is not None and cancel_token.is_cancelled():
                return False
            time.sleep(max(interval, 0.01))
        return True


class InferenceCancelled(Exception):
    """Raised by callers that want to distinguish user cancellation."""
