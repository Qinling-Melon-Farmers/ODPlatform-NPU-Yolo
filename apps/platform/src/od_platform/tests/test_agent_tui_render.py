import unittest

from od_platform.agent.orchestrator import AgentEvent
from od_platform.agent.tools import ToolResult
from od_platform.agent.tui.render import render_event, render_user_message


class TestTuiRender(unittest.TestCase):
    def test_render_message(self) -> None:
        line = render_event(AgentEvent(kind="message", content="你好"))
        self.assertIn("你好", line)
        self.assertIn("🤖", line)

    def test_render_tool_start(self) -> None:
        line = render_event(AgentEvent(kind="tool_start", tool_name="list_datasets", content="{}"))
        self.assertIn("list_datasets", line)
        self.assertIn("{}", line)

    def test_render_tool_result_success(self) -> None:
        event = AgentEvent(
            kind="tool_result",
            tool_name="odp-train",
            tool_result=ToolResult("odp-train", {}, True, 0, "训练完成 mAP50=0.99", ""),
        )
        line = render_event(event)
        self.assertIn("✓", line)
        self.assertIn("mAP50=0.99", line)

    def test_render_tool_result_failed_with_confirmation(self) -> None:
        event = AgentEvent(
            kind="tool_result",
            tool_name="odp-train",
            tool_result=ToolResult(
                "odp-train",
                {},
                False,
                None,
                "需要用户确认",
                "",
                requires_user_action=True,
            ),
        )
        line = render_event(event)
        self.assertIn("✗", line)
        self.assertIn("需确认", line)

    def test_render_error_and_done(self) -> None:
        error_line = render_event(AgentEvent(kind="error", content="API 失败"))
        self.assertIn("API 失败", error_line)
        self.assertIn("❌", error_line)
        done_line = render_event(AgentEvent(kind="done", content="完成"))
        self.assertIn("完成", done_line)

    def test_render_unknown_kind_returns_empty(self) -> None:
        self.assertEqual(render_event(AgentEvent(kind="mystery")), "")

    def test_render_user_message(self) -> None:
        line = render_user_message("用最小模型训练")
        self.assertIn("用最小模型训练", line)
        self.assertIn("你:", line)


if __name__ == "__main__":
    unittest.main()
