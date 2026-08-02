"""Agent 对话编排：ReAct 工具调用循环。

循环语义：系统提示 + 用户消息 → LLM 返回 tool_calls 则逐条执行并把
观察结果（tool role）回填 → LLM 返回纯文本则作为最终回答；
迭代上限耗尽后追加一轮"直接总结"（不带工具）。

@FileName:   orchestrator.py
@Function:   ReAct 循环、事件流与迭代控制
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from od_platform.agent.client import APIError, OpenAIClient
from od_platform.agent.session import AgentSession
from od_platform.agent.tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = """你是 ODPlatform 的目标检测开发平台助手，通过工具完成 YOLO 工作流任务。

工作区约定:
- 原始数据集: data/raw/<dataset>/images/（图片）、data/raw/<dataset>/annotations/（YOLO txt）
- 数据配置: apps/platform/configs/datasets/<name>.yaml
- 模型: 可用模型见 list_available_models / odp-list-models；本地权重在 models/ 下
- 标准流程: 导入(odp-import-dataset) → 转换(odp-transform) → 质检(odp-validate)
  → 训练(odp-train) → 评估(odp-val) → 推理(odp-infer)
- 自动标注: odp-auto-annotate 用 VLM 按自然语言指令批量标注

行动原则:
- 动手前先调用 list_datasets / list_available_models 确认现状，不要猜测数据集或模型名
- 一次只调用一个工具，等结果返回后再决定下一步
- 参数缺失时向用户询问，不要自行编造
- 工具返回非零退出码说明失败，根据错误信息调整参数重试
- 训练/推理等耗时工具默认以 dry-run 模式执行；用户明确要求真实执行时再说明
- 训练必须显式给出 epochs/batch/workers/imgsz 等关键参数，不得依赖配置默认值；参数缺失时向用户询问
- 训练或评估完成后，用 list_run_artifacts 检查产物与指标
- 写文件/消耗 API/GPU 长任务类工具需要用户确认；确认方式为交互模式下的 /confirm <工具名> 命令，单次模式下请告知用户无法确认
- 用中文回答，结论里给出关键数值与产物路径"""


@dataclass(frozen=True)
class AgentConfig:
    """Agent 运行配置。

    Attributes:
        model:          模型名（调用方强制传入，如 deepseek-chat）。
        api_key:        API 密钥；缺省读环境变量 OPENAI_API_KEY。
        base_url:       API 基地址（如 DeepSeek https://api.deepseek.com/v1）。
        max_iterations: 最大工具调用轮数。
        temperature:    采样温度。
    """

    model: str
    api_key: str | None = None
    base_url: str | None = None
    max_iterations: int = 8
    temperature: float = 0.0


@dataclass
class AgentEvent:
    """一次 Agent 运行中的事件（供 CLI/TUI/GUI 渲染与历史回放）。

    Attributes:
        kind:        ``message`` / ``tool_start`` / ``tool_result`` / ``done`` / ``error``。
        content:     消息文本或工具参数 JSON。
        tool_name:   工具名（工具事件）。
        tool_result: 工具执行结果（tool_result 事件）。
        event_id:    事件唯一 ID（按轮递增）。
        session_id:  会话 ID（无会话时为空）。
        turn_id:     对话轮次（用户输入计数，从 1 起）。
        timestamp:   ISO 时间戳。
        status:      状态标记（running/success/failed/cancelled 等）。
        duration_ms: 工具执行耗时（工具事件）。
        artifact_paths: 事件关联的产物路径（预留，TUI 产物面板用）。
        requires_user_action: 需要用户确认/授权（TUI/GUI 据此提示）。
        error_code:  错误码（错误事件）。
    """

    kind: str
    content: str | None = None
    tool_name: str | None = None
    tool_result: ToolResult | None = None
    event_id: str = ""
    session_id: str = ""
    turn_id: int = 0
    timestamp: str = ""
    status: str = ""
    duration_ms: int | None = None
    artifact_paths: list[str] = field(default_factory=list)
    requires_user_action: bool = False
    error_code: str | None = None


class AgentOrchestrator:
    """Agent 对话循环。

    Args:
        client:        OpenAI 兼容客户端。
        registry:      工具注册表。
        config:        运行配置。
        system_prompt: 系统提示词（默认使用内置中文提示）。
        session:       可选的会话对象；提供时每轮对话持久化到会话
                       （多轮记忆），缺省为单轮模式（每次重建消息）。
    """

    def __init__(
        self,
        *,
        client: OpenAIClient,
        registry: ToolRegistry,
        config: AgentConfig,
        system_prompt: str = SYSTEM_PROMPT,
        session: AgentSession | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.config = config
        self.system_prompt = system_prompt
        self.session = session
        self._event_counter = 0
        self._turn_counter = 0

    def _event(self, kind: str, **kwargs: Any) -> AgentEvent:
        """构造带元数据（事件 ID/会话/轮次/时间戳）的事件。"""
        self._event_counter += 1
        return AgentEvent(
            kind=kind,
            event_id=str(self._event_counter),
            session_id=self.session.session_id if self.session is not None else "",
            turn_id=self._turn_counter,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **kwargs,
        )

    def run(self, user_message: str) -> list[AgentEvent]:
        """阻塞式执行一轮对话，返回全部事件。"""
        return list(self.run_stream(user_message))

    def run_stream(self, user_message: str) -> Iterator[AgentEvent]:
        """流式执行一轮对话。

        提供 ``session`` 时：本轮消息（用户输入、工具调用与观察、
        最终回答）全部持久化到会话，下一轮可引用历史。

        Yields:
            按发生顺序的事件；最终以 ``done`` 事件结束（携带最终文本）。
        """
        self._turn_counter += 1
        if self.session is not None:
            history = self.session.to_history()
            if not history or history[0].get("role") != "system":
                history.insert(0, {"role": "system", "content": self.system_prompt})
            messages: list[dict[str, Any]] = [*history, {"role": "user", "content": user_message}]
            start_index = len(history)
        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ]
            start_index = 1

        final_text = ""
        for _iteration in range(self.config.max_iterations):
            try:
                response = self.client.chat(
                    messages,
                    model=self.config.model,
                    tools=self.registry.schemas(),
                    temperature=self.config.temperature,
                )
            except APIError as exc:
                logger.error("Agent API 调用失败: %s", exc)
                yield self._event(
                    "error",
                    content=str(exc),
                    error_code=f"api_error_{exc.status}" if exc.status else "api_error",
                )
                return

            choices = response.get("choices") or []
            if not choices:
                raise APIError(0, "API 返回畸形响应: 缺少 choices")
            assistant_message = choices[0].get("message") or {}
            # 规范化 content（个别严格实现要求 assistant 消息 content 为字符串）
            messages.append({**assistant_message, "content": assistant_message.get("content") or ""})

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                final_text = assistant_message.get("content") or ""
                if final_text:
                    yield self._event("message", content=final_text, status="success")
                break

            for call in tool_calls:
                function = call.get("function") or {}
                name = function.get("name") or ""
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    logger.warning("工具 %s 参数不是合法 JSON，按空参数处理", name)
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}

                yield self._event(
                    "tool_start",
                    tool_name=name,
                    content=json.dumps(arguments, ensure_ascii=False),
                    status="running",
                )
                started = time.perf_counter()
                result = self.registry.execute(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") or "",
                        "content": result.summary,
                    }
                )
                yield self._event(
                    "tool_result",
                    tool_name=name,
                    tool_result=result,
                    status="success" if result.ok else "failed",
                    duration_ms=result.duration_ms
                    or int((time.perf_counter() - started) * 1000),
                    requires_user_action=result.requires_user_action,
                )
        else:
            # 迭代上限耗尽：追加不带工具的"直接总结"轮
            logger.warning("达到最大迭代次数 %d，请求模型直接总结", self.config.max_iterations)
            try:
                response = self.client.chat(
                    messages,
                    model=self.config.model,
                    temperature=self.config.temperature,
                )
            except APIError as exc:
                logger.error("Agent 总结轮 API 调用失败: %s", exc)
                yield self._event(
                    "error",
                    content=str(exc),
                    error_code=f"api_error_{exc.status}" if exc.status else "api_error",
                )
                return
            choices = response.get("choices") or []
            if not choices:
                yield self._event("error", content="API 返回畸形响应: 缺少 choices", error_code="malformed_response")
                return
            summary_message = choices[0].get("message") or {}
            final_text = summary_message.get("content") or "（模型未返回总结文本）"
            messages.append({**summary_message, "content": final_text})
            yield self._event("message", content=final_text, status="success")

        if self.session is not None:
            for message in messages[start_index:]:
                self.session.append(message)
            self.session.save()

        yield self._event("done", content=final_text, status="success")
