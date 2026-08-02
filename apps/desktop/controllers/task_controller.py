"""CLI 任务控制器：CommandWorker 线程生命周期（镜像 AgentController）。"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, Slot
from task_worker import CommandWorker


class TaskController(QObject):
    """管理 CLI 任务子进程线程。

    Signals:
        output_ready: CLI 输出行（合并 stdout+stderr）。
        finished:     任务退出码。
        failed:       任务失败信息。
        busy_changed: 是否运行中（控制按钮状态）。
    """

    output_ready = Signal(str)
    finished = Signal(int)
    failed = Signal(str)
    busy_changed = Signal(bool)

    def __init__(self, cwd: str, platform_src: str) -> None:
        super().__init__()
        self._cwd = cwd
        self._platform_src = platform_src
        self._thread: QThread | None = None
        self._worker: CommandWorker | None = None

    @Slot()
    def start(self, module: str, args: list[str]) -> None:
        """启动 CLI 任务（防重入）。"""
        if self._thread is not None:
            return
        self._thread = QThread(self)
        self._worker = CommandWorker(
            module=module,
            args=args,
            cwd=self._cwd,
            platform_src=self._platform_src,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.output_ready.connect(self.output_ready)
        self._worker.completed.connect(self.finished)
        self._worker.failed.connect(self.failed)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()
        self.busy_changed.emit(True)

    @Slot()
    def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    @Slot()
    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.busy_changed.emit(False)
