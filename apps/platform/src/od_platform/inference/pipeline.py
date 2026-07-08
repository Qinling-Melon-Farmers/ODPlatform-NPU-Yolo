"""Frame-by-frame inference pipeline."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from od_platform.frame_source import (
    BufferStrategy,
    CameraConfig,
    SourceType,
    create_frame_source,
    detect_source_type,
)
from od_platform.inference.cancel import CancelToken
from od_platform.inference.hooks import FrameEvent, InferHooks, ProgressEvent
from od_platform.inference.overlay import Metrics, draw_hud
from od_platform.inference.sinks import OutputSink
from od_platform.visualization import Detection

logger = logging.getLogger(__name__)

_STOP = object()


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
    source_mode: str = "sequential"
    source_buffer: str | None = None
    pipeline_stages: list[str] = field(default_factory=list)

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
            "source_mode": self.source_mode,
            "source_buffer": self.source_buffer,
            "pipeline_stages": list(self.pipeline_stages),
        }


@dataclass
class _InferPacket:
    frame: Any
    result: Any
    detections: list[Detection]
    elapsed_seconds: float


@dataclass
class _RenderPacket:
    frame: Any
    result: Any
    detections: list[Detection]
    annotated: Any
    infer_seconds: float
    render_seconds: float


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
        threaded: bool = False,
        source_buffer: BufferStrategy | None = None,
        buffer_size: int = 8,
        read_timeout: float = 5.0,
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
        self.threaded = threaded
        self.source_buffer = source_buffer
        self.buffer_size = max(1, buffer_size)
        self.read_timeout = max(0.1, read_timeout)
        self.hooks = hooks or InferHooks()
        self.cancel_token = cancel_token
        self.max_frames = max_frames

    def run(self) -> InferStats:
        if self.threaded:
            return self._run_staged()
        return self._run_sequential()

    def _run_sequential(self) -> InferStats:
        metrics = Metrics()
        stats = InferStats()
        start = time.perf_counter()
        last_loop = start
        opened_sink = False

        try:
            with self._create_source() as source:
                stats.source_mode = "sequential"
                stats.source_buffer = self._resolved_source_buffer()
                stats.pipeline_stages = ["read+infer+render+output"]
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

    def _run_staged(self) -> InferStats:
        metrics = Metrics()
        stats = InferStats(source_mode="threaded", source_buffer=self._resolved_source_buffer())
        stats.pipeline_stages = ["read", "infer", "render", "output"]
        start = time.perf_counter()
        last_loop = start
        stop_event = threading.Event()
        errors: list[BaseException] = []
        frame_queue: queue.Queue[Any] = queue.Queue(
            maxsize=1 if stats.source_buffer == "latest" else self.buffer_size
        )
        infer_queue: queue.Queue[Any] = queue.Queue(maxsize=self.buffer_size)
        render_queue: queue.Queue[Any] = queue.Queue(maxsize=self.buffer_size)
        opened_sink = False

        source_type = detect_source_type(self.source)
        self.output_sink.open(self.output_dir, source_type)
        opened_sink = True

        def cancelled() -> bool:
            return stop_event.is_set() or self._cancelled()

        def remember_error(exc: BaseException) -> None:
            errors.append(exc)
            stop_event.set()

        def put_item(target: queue.Queue[Any], item: Any, *, latest: bool = False) -> None:
            if latest:
                self._drain_queue(target)
                try:
                    target.put_nowait(item)
                    return
                except queue.Full:
                    return
            while not cancelled():
                try:
                    target.put(item, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def put_stop(target: queue.Queue[Any], *, clear: bool = False) -> None:
            if clear:
                self._drain_queue(target)
            while True:
                try:
                    target.put(_STOP, timeout=0.1)
                    return
                except queue.Full:
                    if clear or stop_event.is_set():
                        self._drain_queue(target)

        def read_worker() -> None:
            try:
                with create_frame_source(self.source, self.camera_config, stride=self.stride) as source:
                    for frame in source:
                        if cancelled():
                            break
                        if frame.info.frame_index < self.warmup_frames:
                            continue
                        put_item(frame_queue, frame, latest=stats.source_buffer == "latest")
            except BaseException as exc:  # pragma: no cover - defensive around device/codec backends.
                remember_error(exc)
            finally:
                put_stop(frame_queue, clear=stats.source_buffer == "latest")

        def infer_worker() -> None:
            try:
                while not cancelled():
                    item = self._get_queue_item(frame_queue, stop_event)
                    if item is None:
                        continue
                    if item is _STOP:
                        break
                    infer_start = time.perf_counter()
                    result, detections = self.processor.infer(item.image)
                    elapsed = time.perf_counter() - infer_start
                    put_item(infer_queue, _InferPacket(item, result, detections, elapsed))
            except BaseException as exc:  # pragma: no cover - model backend errors are covered by service fallback.
                remember_error(exc)
            finally:
                put_stop(infer_queue)

        def render_worker() -> None:
            try:
                while not cancelled():
                    item = self._get_queue_item(infer_queue, stop_event)
                    if item is None:
                        continue
                    if item is _STOP:
                        break
                    render_start = time.perf_counter()
                    annotated = self.processor.draw(item.frame.image, item.result, item.detections)
                    render_elapsed = time.perf_counter() - render_start
                    put_item(
                        render_queue,
                        _RenderPacket(
                            frame=item.frame,
                            result=item.result,
                            detections=item.detections,
                            annotated=annotated,
                            infer_seconds=item.elapsed_seconds,
                            render_seconds=render_elapsed,
                        ),
                    )
            except BaseException as exc:  # pragma: no cover - rendering backends are environment-dependent.
                remember_error(exc)
            finally:
                put_stop(render_queue)

        workers = [
            threading.Thread(target=read_worker, name="odp-infer-read", daemon=True),
            threading.Thread(target=infer_worker, name="odp-infer-infer", daemon=True),
            threading.Thread(target=render_worker, name="odp-infer-render", daemon=True),
        ]
        for worker in workers:
            worker.start()

        try:
            while not stop_event.is_set():
                item = self._get_queue_item(render_queue, stop_event, timeout=0.1)
                if item is None:
                    if errors:
                        raise errors[0]
                    continue
                if item is _STOP:
                    break

                metrics.infer.update(item.infer_seconds * 1000.0)
                metrics.render.update(item.render_seconds * 1000.0)
                metrics.add_speed(getattr(item.result, "speed", None))

                self.output_sink.write(item.frame, item.annotated)
                if self.show:
                    self._show_frame(item.annotated, metrics, len(item.detections))

                stats.frames += 1
                stats.inference_seconds += item.infer_seconds
                stats.record_detections(item.detections)
                self.hooks.fire_frame(
                    FrameEvent(
                        frame_index=item.frame.info.frame_index,
                        image=item.frame.image,
                        annotated=item.annotated,
                        detections=[_detection_to_dict(det) for det in item.detections],
                    )
                )
                if (
                    self.hooks.on_progress is not None
                    and stats.frames % max(1, self.hooks.progress_interval_frames) == 0
                ):
                    self.hooks.fire_progress(
                        ProgressEvent(
                            frame_index=stats.frames,
                            total_frames=item.frame.info.total_frames,
                            elapsed_seconds=time.perf_counter() - start,
                            loop_fps=metrics.loop.fps,
                            detections_total=stats.detections,
                        )
                    )

                loop_now = time.perf_counter()
                metrics.loop.update((loop_now - last_loop) * 1000.0)
                last_loop = loop_now

                if self.max_frames is not None and stats.frames >= self.max_frames:
                    stats.interrupted = True
                    stop_event.set()
                    break
                if self._cancelled():
                    stats.interrupted = True
                    stop_event.set()
                    break

            if errors:
                raise errors[0]
        finally:
            stop_event.set()
            for target in (frame_queue, infer_queue, render_queue):
                put_stop(target, clear=True)
            for worker in workers:
                worker.join(timeout=max(self.read_timeout, 0.5))
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
        logger.info(
            "staged inference pipeline finished: frames=%s detections=%s",
            stats.frames,
            stats.detections,
        )
        return stats

    def _cancelled(self) -> bool:
        return self.cancel_token is not None and self.cancel_token.is_cancelled()

    def _create_source(self):
        return create_frame_source(self.source, self.camera_config, stride=self.stride)

    def _resolved_source_buffer(self) -> BufferStrategy | None:
        if not self.threaded:
            return None
        if self.source_buffer is not None:
            return self.source_buffer
        source_type = detect_source_type(self.source)
        if source_type == SourceType.CAMERA:
            return "latest"
        return "bounded"

    @staticmethod
    def _drain_queue(target: queue.Queue[Any]) -> None:
        while True:
            try:
                target.get_nowait()
            except queue.Empty:
                return

    @staticmethod
    def _get_queue_item(
        target: queue.Queue[Any],
        stop_event: threading.Event,
        *,
        timeout: float = 0.1,
    ) -> Any:
        while not stop_event.is_set():
            try:
                return target.get(timeout=timeout)
            except queue.Empty:
                return None
        return _STOP

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
