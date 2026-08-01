import json
import unittest

from od_platform.agent.client import APIError
from od_platform.agent.orchestrator import AgentConfig, AgentOrchestrator
from od_platform.agent.tools import ToolRegistry


class FakeClient:
    """按预设响应序列返回 chat 结果的假客户端。"""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def chat(self, messages, *, model, tools=None, temperature=0.0, **kwargs) -> dict:
        self.calls.append({"model": model, "tools": tools, "messages": messages})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _text_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _tool_response(calls: list[dict]) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls}}]}


def _make_orchestrator(client: FakeClient, registry: ToolRegistry | None = None) -> AgentOrchestrator:
    config = AgentConfig(model="test-model", api_key="k", base_url="https://example.com/v1", max_iterations=3)
    return AgentOrchestrator(client=client, registry=registry or ToolRegistry(), config=config)


class TestAgentOrchestrator(unittest.TestCase):
    def test_loop_tool_calls_then_final_text(self) -> None:
        """工具调用后回填观察，最终返回文本。"""
        registry = ToolRegistry()
        called: list[str] = []

        @registry.register_service("hello", description="测试工具")
        def _hello(arguments: dict) -> object:
            called.append(arguments["name"])
            from od_platform.agent.tools import ToolResult

            return ToolResult("hello", arguments, True, None, f"你好 {arguments['name']}", "")

        client = FakeClient(
            [
                _tool_response(
                    [
                        {
                            "id": "call_1",
                            "function": {"name": "hello", "arguments": json.dumps({"name": "小明"})},
                        }
                    ]
                ),
                _text_response("完成"),
            ]
        )
        orchestrator = _make_orchestrator(client, registry)
        events = orchestrator.run("打招呼")

        kinds = [event.kind for event in events]
        self.assertEqual(kinds, ["tool_start", "tool_result", "message", "done"])
        self.assertEqual(called, ["小明"])
        self.assertEqual(events[-1].content, "完成")
        # 工具观察已回填：第二轮消息含 tool role
        self.assertEqual(client.calls[1]["model"], "test-model")

    def test_tool_failure_observation_fed_back(self) -> None:
        registry = ToolRegistry()

        @registry.register_service("broken", description="失败工具")
        def _broken(_arguments: dict) -> object:
            from od_platform.agent.tools import ToolResult

            return ToolResult("broken", {}, False, 2, "参数缺失", "")

        client = FakeClient(
            [
                _tool_response([{"id": "c1", "function": {"name": "broken", "arguments": "{}"}}]),
                _text_response("已重试"),
            ]
        )
        events = _make_orchestrator(client, registry).run("执行")
        tool_event = next(event for event in events if event.kind == "tool_result")
        self.assertFalse(tool_event.tool_result.ok if tool_event.tool_result else True)
        self.assertIn("参数缺失", tool_event.tool_result.summary if tool_event.tool_result else "")

    def test_max_iterations_exhausted_forces_summary(self) -> None:
        client = FakeClient(
            [
                _tool_response([{"id": "c1", "function": {"name": "no-such-tool", "arguments": "{}"}}]),
                _tool_response([{"id": "c2", "function": {"name": "no-such-tool", "arguments": "{}"}}]),
                _tool_response([{"id": "c3", "function": {"name": "no-such-tool", "arguments": "{}"}}]),
                _text_response("迭代太多，直接总结"),
            ]
        )
        events = _make_orchestrator(client).run("循环")
        kinds = [event.kind for event in events]
        self.assertIn("done", kinds)
        self.assertEqual(events[-1].content, "迭代太多，直接总结")

    def test_api_error_emits_error_event(self) -> None:
        client = FakeClient([APIError(429, "限流")])
        events = _make_orchestrator(client).run("测试")
        self.assertEqual(events[0].kind, "error")
        self.assertIn("限流", events[0].content or "")

    def test_assistant_message_content_normalized(self) -> None:
        """带 tool_calls 的 assistant 消息 content=None 应规范化为空字符串。"""
        client = FakeClient(
            [
                _tool_response([{"id": "c1", "function": {"name": "no-such-tool", "arguments": "{}"}}]),
                _text_response("完成"),
            ]
        )
        _make_orchestrator(client).run("执行")
        # 第二轮请求的 messages 中 assistant 消息 content 必须为字符串
        second_call = client.calls[1]
        assistant_msg = next(
            msg for msg in second_call["messages"] if msg.get("role") == "assistant" and msg.get("tool_calls")
        )
        self.assertEqual(assistant_msg["content"], "")

    def test_invalid_tool_arguments_treated_as_empty(self) -> None:
        registry = ToolRegistry()
        received: list[dict] = []

        @registry.register_service("capture", description="捕获参数")
        def _capture(arguments: dict) -> object:
            received.append(arguments)
            from od_platform.agent.tools import ToolResult

            return ToolResult("capture", arguments, True, None, "ok", "")

        client = FakeClient(
            [
                _tool_response([{"id": "c1", "function": {"name": "capture", "arguments": "not-json"}}]),
                _text_response("完成"),
            ]
        )
        _make_orchestrator(client, registry).run("执行")
        self.assertEqual(received, [{}])


if __name__ == "__main__":
    unittest.main()
