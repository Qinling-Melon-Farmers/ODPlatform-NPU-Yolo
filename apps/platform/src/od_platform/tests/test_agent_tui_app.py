"""TUI App 冒烟测试（textual pilot）。

验证：提交消息后 worker 在线程执行、事件经 call_from_thread 渲染
（回归：run_worker 误传调用结果导致主线程立即执行的问题）。

textual 为软依赖：缺失时跳过。
"""

import unittest

try:
    import textual  # noqa: F401
except ImportError:
    textual = None  # type: ignore[assignment]

from od_platform.agent.orchestrator import AgentConfig
from od_platform.agent.tools import ToolRegistry
from od_platform.agent.tui.app import ODPAgentApp


class FakeClient:
    """不调 API 的假客户端。"""

    def chat(self, messages, *, model, tools=None, temperature=0.0, **kwargs) -> dict:
        return {"choices": [{"message": {"role": "assistant", "content": "测试回答"}}]}


@unittest.skipIf(textual is None, "textual 未安装")
class TestTuiAppSmoke(unittest.TestCase):
    def test_submit_renders_conversation(self) -> None:
        import asyncio

        from textual.widgets import Input

        config = AgentConfig(model="m", api_key="k", base_url="https://x/v1")
        app = ODPAgentApp(client=FakeClient(), registry=ToolRegistry(), config=config)

        async def _run() -> None:
            async with app.run_test() as pilot:
                app.on_input_submitted(Input.Submitted(Input(), value="hello"))
                for _ in range(30):
                    await pilot.pause(0.1)
                conversation = app.query_one("#conversation")
                text = "\n".join(line.text for line in conversation.lines)
                self.assertIn("hello", text)
                self.assertIn("测试回答", text)

        asyncio.run(_run())

    def test_slash_command_sessions(self) -> None:
        import asyncio

        from textual.widgets import Input

        config = AgentConfig(model="m", api_key="k", base_url="https://x/v1")
        app = ODPAgentApp(client=FakeClient(), registry=ToolRegistry(), config=config)

        async def _run() -> None:
            async with app.run_test() as pilot:
                app.on_input_submitted(Input.Submitted(Input(), value="/help"))
                for _ in range(10):
                    await pilot.pause(0.1)
                conversation = app.query_one("#conversation")
                text = "\n".join(line.text for line in conversation.lines)
                self.assertIn("/sessions", text)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
