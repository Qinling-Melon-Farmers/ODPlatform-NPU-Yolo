"""Frame-by-frame inference pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from od_platform.frame_source import CameraConfig, create_frame_source
from od_platform.inference.cancel import CancelToken
from od_platform.inference.hooks import FrameEvent, InferHooks, ProgressEvent
from od_platform.inference.overlay import Metrics, draw_hud
from od_platform.inference.sinks import OutputSink
from od_platform.visualization import Detection

logger = logging.getLogger(__name__)


@dataclass
class InferStats:
    """Statistics collected during one inference run."""

    frames: int = 0
    detections: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    wall_seconds: float = 0.0
    inference_seconds: float = 0.0
    interrupted: bool = False
    fps: dict[str, float] = field(default_factory=dict)
    speed_ms: dict[str, float] = field(default_factory=dict)

    @property
    def avg_fps(self) -> float:
        return self.frames / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.inference_seconds / self.frames * 1000.0 if self.frames else 0.0

    def record_detections(self, detections: list[Detection]) -> None:
        self.detections += len(detections)
        for detection in detections:
            self.per_class[detection.label] = self.per_class.get(detection.label, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "detections": self.detections,
            "per_class": dict(sorted(self.per_class.items())),
            "wall_seconds": round(self.wall_seconds, 4),
            "inference_seconds": round(self.inference_seconds, 4),
            "avg_fps": round(self.avg_fps, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "interrupted": self.interrupted,
            "fps": self.fps,
            "speed_ms": self.speed_ms,
        }


class SequentialInferencePipeline:
    """Sequential baseline pipeline for camera, image, folder and video sources."""

    def __init__(
        self,
        *,
        processor: Any,
        source: str | Path | int,
        camera_config: CameraConfig | None,
        output_dir: Path,
        output_sink: OutputSink,
        save: bool,
        show: bool,
        show_info: bool,
        window_name: str,
        warmup_frames: int = 0,
        stride: int = 1,
        hooks: InferHooks | None = None,
        cancel_token: CancelToken | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.processor = processor
        self.source = source
        self.camera_config = camera_config
        self.output_dir = output_dir
        self.output_sink = output_sink
        self.save = save
        self.show = show
        self.show_info = show_info
        self.window_name = window_name
        self.warmup_frames = max(0, warmup_frames)
        self.stride = max(1, stride)
        self.hooks = hooks or InferHooks()
        self.cancel_token = cancel_token
        self.max_frames = max_frames

    def run(self) -> InferStats:
        metrics = Metrics()
        stats = InferStats()
        start = time.perf_counter()
        last_loop = start
        opened_sink = False

        try:
            with create_frame_source(self.source, self.camera_config, stride=self.stride) as source:
                self.output_sink.open(self.output_dir, source.get_source_type())
                opened_sink = True
                for frame in source:
                    if self._cancelled():
                        stats.interrupted = True
                        break
                    if frame.info.frame_index < self.warmup_frames:
                        continue
                    if self.max_frames is not None and stats.frames >= self.max_frames:
                        break

                    capture_now = time.perf_counter()
                    metrics.capture.update((capture_now - last_loop) * 1000.0)

                    infer_start = time.perf_counter()
                    result, detections = self.processor.infer(frame.image)
                    infer_elapsed = time.perf_counter() - infer_start
                    stats.inference_seconds += infer_elapsed
                    metrics.infer.update(infer_elapsed * 1000.0)
                    metrics.add_speed(getattr(result, "speed", None))

                    render_start = time.perf_counter()
                    annotated = self.processor.draw(frame.image, result, detections)
                    metrics.render.update((time.perf_counter() - render_start) * 1000.0)

                    self.output_sink.write(frame, annotated)
                    if self.show:
                        self._show_frame(annotated, metrics, len(detections))

                    stats.frames += 1
                    stats.record_detections(detections)
                    self.hooks.fire_frame(
                        FrameEvent(
                            frame_index=frame.info.frame_index,
                            image=frame.image,
                            annotated=annotated,
                            detections=[_detection_to_dict(item) for item in detections],
                        )
                    )
                    if (
                        self.hooks.on_progress is not None
                        and stats.frames % max(1, self.hooks.progress_interval_frames) == 0
                    ):
                        self.hooks.fire_progress(
                            ProgressEvent(
                                frame_index=stats.frames,
                                total_frames=frame.info.total_frames,
                                elapsed_seconds=time.perf_counter() - start,
                                loop_fps=metrics.loop.fps,
                                detections_total=stats.detections,
                            )
                        )

                    loop_now = time.perf_counter()
                    metrics.loop.update((loop_now - last_loop) * 1000.0)
                    last_loop = loop_now
        finally:
            if opened_sink:
                self.output_sink.close()
            if self.show:
                self._close_window()

        stats.wall_seconds = time.perf_counter() - start
        metric_snapshot = metrics.snapshot()
        stats.fps = {
            key: metric_snapshot[key]
            for key in ("capture_fps", "infer_fps", "render_fps", "loop_fps", "current_fps")
        }
        stats.speed_ms = metric_snapshot["speed_ms"]
        logger.info("inference pipeline finished: frames=%s detections=%s", stats.frames, stats.detections)
        return stats

    def _cancelled(self) -> bool:
        return self.cancel_token is not None and self.cancel_token.is_cancelled()

    def _show_frame(self, annotated, metrics: Metrics, detections: int) -> None:
        import cv2

        display = annotated.copy()
        draw_hud(display, metrics, detections=detections, recording=self.save, show_info=self.show_info)
        cv2.imshow(self.window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            if self.cancel_token is not None:
                self.cancel_token.cancel()
        elif key == ord(" "):
            self._pause_window(display)

    @staticmethod
    def _close_window() -> None:
        import cv2

        cv2.destroyAllWindows()

    def _pause_window(self, display) -> None:
        import cv2

        paused = display.copy()
        height, width = paused.shape[:2]
        text = "PAUSED - SPACE resume, Q/Esc quit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.7, 2)
        x = max(10, (width - text_w) // 2)
        y = max(40, (height + text_h) // 2)
        overlay = paused.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, paused, 0.65, 0, paused)
        cv2.putText(paused, text, (x, y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.window_name, paused)

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                return
            if key in (ord("q"), 27):
                if self.cancel_token is not None:
                    self.cancel_token.cancel()
                return


def _detection_to_dict(detection: Detection) -> dict[str, Any]:
    return {
        "box": list(detection.box),
        "confidence": detection.confidence,
        "label": detection.label,
        "color": list(detection.color),
    }
