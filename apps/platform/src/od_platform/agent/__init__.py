"""AI Agent 子系统：LLM 工具化执行目标检测任务。

分层：``client`` 为通用 OpenAI 兼容客户端（叶子模块，VLM 标注复用），
``schema`` 将 CLI 自省为 tool schema，``tools`` 提供工具注册与执行，
``orchestrator`` 实现 ReAct 对话循环。后续层次实现后逐步补全导出。
"""

from __future__ import annotations

from od_platform.agent.client import (
    APIError,
    OpenAIClient,
    encode_image_base64,
    image_url_content_part,
)

__all__ = [
    "APIError",
    "OpenAIClient",
    "encode_image_base64",
    "image_url_content_part",
]
