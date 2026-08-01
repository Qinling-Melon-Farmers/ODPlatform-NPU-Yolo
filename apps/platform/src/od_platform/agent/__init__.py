"""AI Agent 子系统：LLM 工具化执行目标检测任务。

分层：``client`` 为通用 OpenAI 兼容客户端（叶子模块，VLM 标注复用），
``schema`` 将 CLI 自省为 tool schema，``tools`` 提供工具注册与执行，
``orchestrator`` 实现 ReAct 对话循环。
"""

from __future__ import annotations

from od_platform.agent.client import (
    APIError,
    OpenAIClient,
    encode_image_base64,
    image_url_content_part,
)
from od_platform.agent.orchestrator import SYSTEM_PROMPT, AgentConfig, AgentEvent, AgentOrchestrator
from od_platform.agent.session import AgentSession, latest_session, list_sessions
from od_platform.agent.tools import ToolRegistry, ToolResult, build_default_registry

__all__ = [
    "APIError",
    "SYSTEM_PROMPT",
    "AgentConfig",
    "AgentEvent",
    "AgentOrchestrator",
    "AgentSession",
    "OpenAIClient",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "encode_image_base64",
    "image_url_content_part",
    "latest_session",
    "list_sessions",
]
