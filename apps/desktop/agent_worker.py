"""AI 助手后台线程：驱动 Agent 并在 Qt 事件循环中回传事件。"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from od_platform.agent.client import OpenAIClient
from od_platform.agent.orchestrator import AgentConfig, AgentOrchestrator
from od_platform.agent.tools import build_default_registry


class AgentWorker(QObject):
    """在 QThread 中运行一轮 Agent 对话，逐条回传 AgentEvent。

    Signals:
        event_ready: 一个 AgentEvent（message/tool_start/tool_result/error/done）。
        completed:   最终回答文本（对话自然结束）。
        failed:      致命错误信息（API 配置错误等）。
    """

    event_ready = Signal(object)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str,
        message: str,
        max_iterations: int = 8,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._message = message
        self._max_iterations = max_iterations
        self._dry_run = dry_run
        self._cancelled = False

    @Slot()
    def run(self) -> None:
        """执行一轮 Agent 对话，逐条 emit 事件后以 completed/failed 收尾。"""
        try:
            client = OpenAIClient(api_key=self._api_key, base_url=self._base_url)
            config = AgentConfig(
                model=self._model,
                api_key=self._api_key,
                base_url=self._base_url,
                max_iterations=self._max_iterations,
            )
            registry = build_default_registry(dry_run=self._dry_run)
            orchestrator = AgentOrchestrator(client=client, registry=registry, config=config)

            final_text = ""
            for event in orchestrator.run_stream(self._message):
                if self._cancelled:
                    self.failed.emit("已取消")
                    return
                self.event_ready.emit(event)
                if event.kind == "done":
                    final_text = event.content or ""
        except Exception as exc:  # noqa: BLE001 - GUI boundary must not crash the process.
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.completed.emit(final_text)

    @Slot()
    def cancel(self) -> None:
        """请求取消（当前工具调用完成后停止）。"""
        self._cancelled = True
