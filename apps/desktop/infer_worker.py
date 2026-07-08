"""Background inference worker for the desktop demo."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal, Slot
from qt_sink import QtSignalSink

from od_platform.inference import CancelToken, InferHooks, InferResult, infer_yolo


class InferWorker(QObject):
    """Run platform inference in a QThread and report frames through signals."""

    frame_ready = Signal(object, object)
    progress_ready = Signal(object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        model: str,
        source: str,
        conf: float,
        iou: float,
        device: str | None,
        name: str,
        task: str = "detect",
        imgsz: int | None = None,
        max_det: int | None = None,
        classes: str | None = None,
        runtime_config: str | None = None,
        pipeline_yaml: str | None = None,
        save_outputs: bool = False,
        threaded: bool = True,
        max_frames: int | None = None,
    ) -> None:
        super().__init__()
        self._cancel_token = CancelToken()
        self._model = model
        self._source = source
        self._conf = conf
        self._iou = iou
        self._device = device
        self._name = name
        self._task = task
        self._imgsz = imgsz
        self._max_det = max_det
        self._classes = classes
        self._runtime_config = runtime_config
        self._pipeline_yaml = pipeline_yaml
        self._save_outputs = save_outputs
        self._threaded = threaded
        self._max_frames = max_frames

    @Slot()
    def run(self) -> None:
        """Execute inference until the source ends or cancellation is requested."""
        try:
            result = self._run_inference()
        except Exception as exc:  # noqa: BLE001 - GUI boundary must not crash the process.
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return

        if result.success:
            self.completed.emit(result)
        else:
            self.failed.emit(result.error or "推理失败，未返回详细错误")

    @Slot()
    def cancel(self) -> None:
        """Request graceful cancellation."""
        self._cancel_token.cancel()

    def _run_inference(self) -> InferResult:
        cli_args: dict[str, Any] = {
            "model": self._model,
            "source": self._source,
            "task": self._task,
            "conf": self._conf,
            "iou": self._iou,
            "name": self._name,
            "save": self._save_outputs,
            "show": False,
        }
        if self._device:
            cli_args["device"] = self._device
        if self._imgsz:
            cli_args["imgsz"] = self._imgsz
        if self._max_det:
            cli_args["max_det"] = self._max_det
        if self._classes:
            cli_args["classes"] = [int(item.strip()) for item in self._classes.split(",") if item.strip()]

        sink = QtSignalSink(self.frame_ready.emit, save_to_disk=self._save_outputs)
        hooks = InferHooks(
            on_progress=self.progress_ready.emit,
            on_error=lambda exc: self.failed.emit(str(exc)),
            progress_interval_frames=10,
        )
        return infer_yolo(
            yaml_path=self._runtime_config,
            pipeline_yaml=self._pipeline_yaml,
            cli_args=cli_args,
            beautify=True,
            show_info=False,
            output_sink=sink,
            hooks=hooks,
            cancel_token=self._cancel_token,
            threaded=self._threaded,
            max_frames=self._max_frames,
        )
