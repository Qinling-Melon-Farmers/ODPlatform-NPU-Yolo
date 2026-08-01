"""Agent 会话持久化与多轮对话历史。

会话存储于 ``runs/agent_sessions/<session_id>/``：
- ``session.json`` — 元数据（id/title/时间戳）
- ``messages.jsonl`` — 追加式消息历史（每行一个 JSON 消息）

历史截断：非 system 消息超过 ``max_messages`` 时丢弃最早的消息，
防止上下文无限增长（DeepSeek 等 128K 上下文下简单截断足够）。

@FileName:   session.py
@Function:   会话创建/加载/追加/截断/列表
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from od_platform.common import paths

logger = logging.getLogger(__name__)

SESSION_ID_FORMAT = "%Y%m%d_%H%M%S"


@dataclass
class AgentSession:
    """一个可恢复的 Agent 对话会话。

    Attributes:
        session_id:  会话 ID（时间戳）。
        title:       会话标题（用户可见）。
        created_at:  创建时间（ISO）。
        messages:    消息历史（首条为 system）。
        max_messages: 非 system 消息上限（超出丢最早）。
    """

    session_id: str
    title: str
    created_at: str
    messages: list[dict] = field(default_factory=list)
    max_messages: int = 40

    # ---- 工厂方法 ----

    @classmethod
    def create(cls, system_prompt: str, *, title: str | None = None) -> AgentSession:
        """创建新会话（含 system 消息）；同一秒内多次创建时 ID 自动递增后缀。"""
        now = datetime.now()
        base_id = now.strftime(SESSION_ID_FORMAT)
        session_id = base_id
        counter = 2
        while (paths.RUNS_DIR / "agent_sessions" / session_id).exists():
            session_id = f"{base_id}-{counter}"
            counter += 1
        return cls(
            session_id=session_id,
            title=title or f"会话 {now.strftime('%m-%d %H:%M')}",
            created_at=now.isoformat(timespec="seconds"),
            messages=[{"role": "system", "content": system_prompt}],
        )

    @classmethod
    def load(cls, session_id: str) -> AgentSession | None:
        """按 ID 加载会话；不存在返回 None。"""
        root = paths.RUNS_DIR / "agent_sessions" / session_id
        meta_path = root / "session.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            messages: list[dict] = []
            messages_path = root / "messages.jsonl"
            if messages_path.exists():
                for line in messages_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        messages.append(json.loads(line))
            return cls(
                session_id=meta.get("session_id", session_id),
                title=meta.get("title", session_id),
                created_at=meta.get("created_at", ""),
                messages=messages,
                max_messages=int(meta.get("max_messages", 40)),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("会话 %s 加载失败: %s", session_id, exc)
            return None

    # ---- 历史管理 ----

    def to_history(self) -> list[dict]:
        """返回当前消息列表（截断后的副本，供 orchestrator 使用）。"""
        return list(self.messages)

    def append(self, message: dict) -> None:
        """追加一条消息并执行截断（保留 system 首条）。"""
        self.messages.append(message)
        non_system = [msg for msg in self.messages if msg.get("role") != "system"]
        overflow = len(non_system) - self.max_messages
        if overflow > 0:
            # 从头删除最早的非 system 消息（保留 system）
            kept: list[dict] = []
            removed = 0
            for msg in self.messages:
                if msg.get("role") == "system":
                    kept.append(msg)
                elif removed < overflow:
                    removed += 1
                else:
                    kept.append(msg)
            self.messages = kept
            logger.debug("会话 %s 截断 %d 条历史消息", self.session_id, removed)

    def clear_history(self) -> None:
        """清空历史（保留 system 消息）。"""
        self.messages = [msg for msg in self.messages if msg.get("role") == "system"]

    # ---- 持久化 ----

    @property
    def root_dir(self) -> Path:
        return paths.RUNS_DIR / "agent_sessions" / self.session_id

    def save(self) -> Path:
        """持久化会话（session.json + messages.jsonl）。"""
        root = self.root_dir
        root.mkdir(parents=True, exist_ok=True)
        meta = {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "max_messages": self.max_messages,
        }
        (root / "session.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = "\n".join(json.dumps(msg, ensure_ascii=False) for msg in self.messages)
        (root / "messages.jsonl").write_text(lines + ("\n" if lines else ""), encoding="utf-8")
        return root


def list_sessions(limit: int = 20) -> list[AgentSession]:
    """列出最近会话（按更新时间倒序）。"""
    root = paths.RUNS_DIR / "agent_sessions"
    if not root.exists():
        return []
    candidates: list[AgentSession] = []
    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        session = AgentSession.load(session_dir.name)
        if session is not None:
            candidates.append(session)
    candidates.sort(key=lambda s: s.session_id, reverse=True)
    return candidates[:limit]


def latest_session() -> AgentSession | None:
    """返回最近一次会话；无会话时返回 None。"""
    sessions = list_sessions(limit=1)
    return sessions[0] if sessions else None
