"""HUD rendering and lightweight runtime metrics for inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


class FPSCounter:
    """Sliding-window FPS counter fed with elapsed milliseconds per frame."""

    def __init__(self, window_size: int = 30) -> None:
        self._samples: deque[float] = deque(maxlen=window_size)
        self._last_ms = 0.0

    def update(self, elapsed_ms: float) -> None:
        if elapsed_ms > 0:
            self._samples.append(elapsed_ms)
            self._last_ms = elapsed_ms

    @property
    def fps(self) -> float:
        if not self._samples:
            return 0.0
        avg_ms = sum(self._samples) / len(self._samples)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    @property
    def instant_fps(self) -> float:
        return 1000.0 / self._last_ms if self._last_ms > 0 else 0.0


@dataclass
class Metrics:
    """Runtime metrics split by capture, inference, render and loop."""

    capture: FPSCounter = field(default_factory=FPSCounter)
    infer: FPSCounter = field(default_factory=FPSCounter)
    render: FPSCounter = field(default_factory=FPSCounter)
    loop: FPSCounter = field(default_factory=FPSCounter)
    _speed_totals: dict[str, float] = field(default_factory=dict)
    _speed_count: int = 0

    def add_speed(self, speed: dict[str, Any] | None) -> None:
        if not speed:
            return
        for key in ("preprocess", "inference", "postprocess"):
            self._speed_totals[key] = self._speed_totals.get(key, 0.0) + float(speed.get(key, 0.0))
        self._speed_count += 1
        inference_ms = speed.get("inference")
        if inference_ms:
            self.infer.update(float(inference_ms))

    def snapshot(self) -> dict[str, Any]:
        speed_ms = {}
        if self._speed_count:
            speed_ms = {
                key: round(value / self._speed_count, 3)
                for key, value in self._speed_totals.items()
            }
        return {
            "capture_fps": round(self.capture.fps, 2),
            "infer_fps": round(self.infer.fps, 2),
            "render_fps": round(self.render.fps, 2),
            "loop_fps": round(self.loop.fps, 2),
            "current_fps": round(self.loop.instant_fps, 2),
            "speed_ms": speed_ms,
        }


def draw_hud(frame, metrics: Metrics, *, detections: int = 0, recording: bool = False, show_info: bool = True) -> None:
    """Draw a small OpenCV HUD on an annotated BGR frame."""
    if not show_info:
        return

    import cv2

    rows = [
        ("Capture", f"{metrics.capture.fps:5.1f} FPS", (0, 200, 255)),
        ("Infer", f"{metrics.infer.fps:5.1f} FPS", (0, 230, 0)),
        ("Render", f"{metrics.render.fps:5.1f} FPS", (255, 120, 220)),
        ("Loop", f"{metrics.loop.fps:5.1f} FPS", (255, 200, 0)),
        ("Current", f"{metrics.loop.instant_fps:5.1f} FPS", (255, 255, 255)),
        ("Objects", str(detections), (120, 220, 255)),
    ]
    x0, y0 = 12, 12
    line_h = 22
    panel_w = 230
    panel_h = 20 + line_h * len(rows)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + 4, y0 + panel_h), (0, 200, 255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = y0 + 24
    for label, value, color in rows:
        cv2.putText(frame, label, (x0 + 14, y), font, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x0 + 100, y), font, 0.5, color, 1, cv2.LINE_AA)
        y += line_h

    if recording:
        height, width = frame.shape[:2]
        cv2.circle(frame, (width - 70, 24), 7, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (width - 56, 30), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
