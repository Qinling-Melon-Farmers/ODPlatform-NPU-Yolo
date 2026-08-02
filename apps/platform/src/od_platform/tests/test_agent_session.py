import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from od_platform.agent.session import AgentSession, latest_session, list_sessions


class TestAgentSession(unittest.TestCase):
    def _patch_runs_dir(self, root: Path):
        return patch("od_platform.agent.session.paths.RUNS_DIR", root)

    def test_create_contains_system_message(self) -> None:
        session = AgentSession.create("系统提示")
        self.assertEqual(session.messages[0]["role"], "system")
        self.assertEqual(session.messages[0]["content"], "系统提示")

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patch_runs_dir(Path(temp_dir)):
                session = AgentSession.create("系统")
                session.append({"role": "user", "content": "你好"})
                session.append({"role": "assistant", "content": "你好！"})
                saved_root = session.save()
                self.assertTrue((saved_root / "session.json").exists())
                self.assertTrue((saved_root / "messages.jsonl").exists())

                loaded = AgentSession.load(session.session_id)
                self.assertIsNotNone(loaded)
                if loaded is not None:
                    self.assertEqual(loaded.session_id, session.session_id)
                    self.assertEqual(len(loaded.messages), 3)
                    self.assertEqual(loaded.messages[1]["content"], "你好")

    def test_load_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patch_runs_dir(Path(temp_dir)):
                self.assertIsNone(AgentSession.load("no-such-session"))

    def test_history_truncation_keeps_system(self) -> None:
        session = AgentSession.create("系统", title="t")
        session.max_messages = 5
        for index in range(10):
            session.append({"role": "user", "content": f"消息 {index}"})
        # system + 最近 5 条
        self.assertEqual(len(session.messages), 6)
        self.assertEqual(session.messages[0]["role"], "system")
        self.assertNotIn({"role": "user", "content": "消息 0"}, session.messages)
        self.assertIn({"role": "user", "content": "消息 9"}, session.messages)

    def test_clear_history_keeps_system(self) -> None:
        session = AgentSession.create("系统")
        session.append({"role": "user", "content": "x"})
        session.clear_history()
        self.assertEqual(len(session.messages), 1)
        self.assertEqual(session.messages[0]["role"], "system")

    def test_list_and_latest_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patch_runs_dir(Path(temp_dir)):
                first = AgentSession.create("系统", title="第一个")
                first.save()
                second = AgentSession.create("系统", title="第二个")
                second.save()

                sessions = list_sessions()
                self.assertEqual(len(sessions), 2)
                # 按 ID 倒序（时间戳越大越新）
                self.assertEqual(sessions[0].session_id, second.session_id)

                latest = latest_session()
                self.assertIsNotNone(latest)
                if latest is not None:
                    self.assertEqual(latest.session_id, second.session_id)

    def test_list_sessions_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._patch_runs_dir(Path(temp_dir)):
                self.assertEqual(list_sessions(), [])
                self.assertIsNone(latest_session())


class TestOrchestratorSession(unittest.TestCase):
    """验证 orchestrator 与 session 的多轮记忆集成。"""

    def test_run_stream_persists_history_across_turns(self) -> None:
        from od_platform.agent.orchestrator import AgentConfig, AgentOrchestrator
        from od_platform.agent.tools import ToolRegistry

        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[list[dict]] = []

            def chat(self, messages, *, model, tools=None, temperature=0.0, **kwargs) -> dict:
                self.calls.append(list(messages))
                # 第二轮起应包含第一轮的历史
                content = f"回答（历史 {len(messages)} 条）"
                return {"choices": [{"message": {"role": "assistant", "content": content}}]}

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("od_platform.agent.session.paths.RUNS_DIR", Path(temp_dir)):
                session = AgentSession.create("系统提示")
                client = FakeClient()
                orchestrator = AgentOrchestrator(
                    client=client,
                    registry=ToolRegistry(),
                    config=AgentConfig(model="m", api_key="k", base_url="https://x/v1"),
                    session=session,
                )
                list(orchestrator.run_stream("第一轮"))
                list(orchestrator.run_stream("第二轮"))

                # 第一轮调用只有 system + user
                self.assertEqual(len(client.calls[0]), 2)
                # 第二轮调用包含第一轮历史（system + user1 + assistant1 + user2）
                second_call = client.calls[1]
                self.assertGreaterEqual(len(second_call), 4)
                self.assertEqual(second_call[-1]["role"], "user")
                self.assertEqual(second_call[-1]["content"], "第二轮")
                # 会话已持久化
                self.assertTrue(session.root_dir.exists())
                loaded = AgentSession.load(session.session_id)
                self.assertIsNotNone(loaded)
                if loaded is not None:
                    self.assertGreaterEqual(len(loaded.messages), 5)


if __name__ == "__main__":
    unittest.main()
