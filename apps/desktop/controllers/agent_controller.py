"""AI 助手控制器：线程管理、会话状态与事件路由。"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, Slot

from agent_worker import AgentWorker
from services.session_store_adapter import SessionStoreAdapter


class AgentController(QObject):
    """管理 AgentWorker 线程与当前会话。

    Signals:
        event_ready:     一个 AgentEvent（消息/工具调用/结果）。
        completed:       最终回答文本。
        cancelled:       用户取消。
        failed:          致命错误。
        session_ready:   多轮对话后的会话对象（用于刷新会话列表）。
        status_changed:  (状态, 详情) 用于状态栏。
        busy_changed:    是否运行中（控制发送/停止按钮）。
    """

    event_ready = Signal(object)
    completed = Signal(str)
    cancelled = Signal()
    failed = Signal(str)
    session_ready = Signal(object)
    status_changed = Signal(str, str)
    busy_changed = Signal(bool)

    def __init__(self, store: SessionStoreAdapter | None = None) -> None:
        super().__init__()
        self._store = store or SessionStoreAdapter()
        self._thread: QThread | None = None
        self._worker: AgentWorker | None = None
        self.session = None

    # ---- 会话 ----

    def new_session(self) -> object:
        self.session = self._store.create_session()
        return self.session

    def load_session(self, session_id: str) -> object | None:
        session = self._store.load_session(session_id)
        if session is not None:
            self.session = session
        return session

    def resume_latest(self) -> object | None:
        session = self._store.resume_latest()
        if session is not None:
            self.session = session
        return session

    def list_sessions(self, limit: int = 20) -> list:
        return self._store.list_sessions(limit=limit)

    # ---- 执行 ----

    @Slot()
    def submit(
        self,
        message: str,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        max_iterations: int,
        dry_run: bool,
    ) -> None:
        """启动一轮 Agent 对话（后台线程）。"""
        if self._thread is not None:
            return

        self._thread = QThread(self)
        self._worker = AgentWorker(
            model=model,
            api_key=api_key,
            base_url=base_url,
            message=message,
            max_iterations=max_iterations,
            dry_run=dry_run,
            session=self.session,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.event_ready.connect(self.event_ready)
        self._worker.completed.connect(self.completed)
        self._worker.cancelled.connect(self.cancelled)
        self._worker.failed.connect(self.failed)
        self._worker.session_ready.connect(self._on_session_ready)
        self._worker.completed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()
        self.busy_changed.emit(True)
        self.status_changed.emit("运行中", f"AI 助手: {model}")

    @Slot()
    def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.status_changed.emit("停止中", "正在停止 AI 助手")

    @Slot(object)
    def _on_session_ready(self, session: object) -> None:
        self.session = session
        self.session_ready.emit(session)

    @Slot()
    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.busy_changed.emit(False)
