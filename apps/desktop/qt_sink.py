"""Qt-friendly inference sink for the desktop demo."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from od_platform.frame_source import SourceType
from od_platform.inference.sinks import LocalFileSink, NullSink, OutputSink


class QtSignalSink(OutputSink):
    """Emit annotated frames to the UI and optionally tee them to disk."""

    def __init__(
        self,
        emit_frame: Callable[[Any, np.ndarray], None],
        *,
        save_to_disk: bool = False,
    ) -> None:
        self._emit_frame = emit_frame
        self._delegate: OutputSink = LocalFileSink() if save_to_disk else NullSink()

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        self._delegate.open(output_dir, source_type)

    def write(self, frame: Any, annotated: np.ndarray) -> None:
        display_frame = annotated.copy()
        self._emit_frame(frame, display_frame)
        self._delegate.write(frame, annotated)

    def close(self) -> None:
        self._delegate.close()
