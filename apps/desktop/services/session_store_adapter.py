"""桌面端 Agent 会话存储适配层（薄封装 od_platform.agent.session）。"""

from __future__ import annotations

from od_platform.agent.session import AgentSession, latest_session, list_sessions


class SessionStoreAdapter:
    """桌面端会话管理：列出/创建/加载/恢复最新。"""

    def __init__(self, system_prompt: str | None = None) -> None:
        self._system_prompt = system_prompt

    def list_sessions(self, limit: int = 20) -> list[AgentSession]:
        return list_sessions(limit=limit)

    def create_session(self) -> AgentSession:
        from od_platform.agent.orchestrator import SYSTEM_PROMPT

        return AgentSession.create(self._system_prompt or SYSTEM_PROMPT)

    def load_session(self, session_id: str) -> AgentSession | None:
        return AgentSession.load(session_id)

    def resume_latest(self) -> AgentSession | None:
        return latest_session()
