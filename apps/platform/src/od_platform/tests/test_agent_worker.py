import sys
import unittest
from pathlib import Path
from unittest.mock import patch

DESKTOP_DIR = Path(__file__).resolve().parents[4] / "desktop"
if str(DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(DESKTOP_DIR))

from agent_worker import AgentWorker  # noqa: E402
from PySide6.QtCore import QCoreApplication  # noqa: E402

from od_platform.agent.orchestrator import AgentEvent  # noqa: E402

_app = QCoreApplication.instance() or QCoreApplication([])


class FakeOrchestrator:
    """按预设事件序列返回的假编排器。"""

    session = None  # 与真实 AgentOrchestrator 对齐

    def __init__(self, events: list[AgentEvent]) -> None:
        self.events = events

    def run_stream(self, _message: str):
        yield from self.events


class TestAgentWorker(unittest.TestCase):
    def _make_worker(self, events: list[AgentEvent]) -> AgentWorker:
        return AgentWorker(
            model="deepseek-chat",
            api_key="k",
            base_url="https://api.deepseek.com/v1",
            message="列出数据集",
        )

    def test_run_emits_events_and_completed(self) -> None:
        events = [
            AgentEvent(kind="tool_start", tool_name="list_datasets", content="{}"),
            AgentEvent(kind="message", content="共 1 个数据集"),
            AgentEvent(kind="done", content="共 1 个数据集"),
        ]
        worker = self._make_worker(events)

        received: list[AgentEvent] = []
        completed: list[str] = []
        failed: list[str] = []
        cancelled: list[int] = []
        worker.event_ready.connect(received.append)
        worker.completed.connect(completed.append)
        worker.failed.connect(failed.append)
        worker.cancelled.connect(lambda: cancelled.append(1))

        with patch("agent_worker.OpenAIClient"), patch(
            "agent_worker.AgentOrchestrator", return_value=FakeOrchestrator(events)
        ):
            worker.run()

        self.assertEqual([event.kind for event in received], ["tool_start", "message", "done"])
        self.assertEqual(completed, ["共 1 个数据集"])
        self.assertEqual(failed, [])
        self.assertEqual(cancelled, [])

    def test_cancel_emits_cancelled_not_failed(self) -> None:
        worker = self._make_worker([AgentEvent(kind="message", content="hi")])

        cancelled: list[int] = []
        failed: list[str] = []
        worker.cancelled.connect(lambda: cancelled.append(1))
        worker.failed.connect(failed.append)

        with patch("agent_worker.OpenAIClient"), patch(
            "agent_worker.AgentOrchestrator", return_value=FakeOrchestrator([AgentEvent(kind="done", content="")])
        ):
            worker.cancel()
            worker.run()

        self.assertEqual(cancelled, [1])
        self.assertEqual(failed, [])

    def test_api_error_emits_failed(self) -> None:
        worker = self._make_worker([])

        failed: list[str] = []
        worker.failed.connect(failed.append)

        class BoomClient:
            def __init__(self, **kwargs):
                raise ValueError("缺少 API 基地址")

        with patch("agent_worker.OpenAIClient", BoomClient):
            worker.run()

        self.assertEqual(len(failed), 1)
        self.assertIn("缺少 API 基地址", failed[0])


if __name__ == "__main__":
    unittest.main()
