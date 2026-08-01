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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from od_platform.agent.client import APIError, OpenAIClient
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
    """一次 Agent 运行中的事件（供 CLI/桌面端渲染）。

    Attributes:
        kind:        ``message`` / ``tool_start`` / ``tool_result`` / ``done`` / ``error``。
        content:     消息文本或工具参数 JSON。
        tool_name:   工具名（工具事件）。
        tool_result: 工具执行结果（tool_result 事件）。
    """

    kind: str
    content: str | None = None
    tool_name: str | None = None
    tool_result: ToolResult | None = None


class AgentOrchestrator:
    """Agent 对话循环。

    Args:
        client:        OpenAI 兼容客户端。
        registry:      工具注册表。
        config:        运行配置。
        system_prompt: 系统提示词（默认使用内置中文提示）。
    """

    def __init__(
        self,
        *,
        client: OpenAIClient,
        registry: ToolRegistry,
        config: AgentConfig,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.registry = registry
        self.config = config
        self.system_prompt = system_prompt

    def run(self, user_message: str) -> list[AgentEvent]:
        """阻塞式执行一轮对话，返回全部事件。"""
        return list(self.run_stream(user_message))

    def run_stream(self, user_message: str) -> Iterator[AgentEvent]:
        """流式执行一轮对话。

        Yields:
            按发生顺序的事件；最终以 ``done`` 事件结束（携带最终文本）。
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

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
                yield AgentEvent(kind="error", content=str(exc))
                return

            choice = response["choices"][0]
            assistant_message = choice["message"]
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                final_text = assistant_message.get("content") or ""
                if final_text:
                    yield AgentEvent(kind="message", content=final_text)
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

                yield AgentEvent(
                    kind="tool_start",
                    tool_name=name,
                    content=json.dumps(arguments, ensure_ascii=False),
                )
                result = self.registry.execute(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id") or "",
                        "content": result.summary,
                    }
                )
                yield AgentEvent(kind="tool_result", tool_name=name, tool_result=result)
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
                yield AgentEvent(kind="error", content=str(exc))
                return
            final_text = response["choices"][0]["message"].get("content") or "（模型未返回总结文本）"
            yield AgentEvent(kind="message", content=final_text)

        yield AgentEvent(kind="done", content=final_text)
