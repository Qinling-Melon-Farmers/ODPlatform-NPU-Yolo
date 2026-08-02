"""Agent 事件 → TUI 行文本渲染（纯函数，不依赖 textual，可直接单测）。

输出使用 textual markup 语法（``[bold]...[/]``），由 RichLog 渲染。

@FileName:   render.py
@Function:   AgentEvent 渲染为 TUI 行
"""

from __future__ import annotations

from od_platform.agent.orchestrator import AgentEvent

#: 工具结果摘要截断长度。
SUMMARY_PREVIEW = 200


def render_event(event: AgentEvent) -> str:
    """将 AgentEvent 渲染为一行 TUI 文本（textual markup）。

    Args:
        event: Agent 事件。

    Returns:
        文本行；未知 kind 返回空字符串。
    """
    if event.kind == "message" and event.content:
        return f"[bold blue]🤖[/] {event.content}"
    if event.kind == "tool_start":
        return f"[yellow]→[/] [bold]{event.tool_name}[/]({event.content or ''})"
    if event.kind == "tool_result":
        result = event.tool_result
        if result is None:
            return f"[dim]{event.tool_name} 结果缺失[/]"
        mark = "✓" if result.ok else "✗"
        color = "green" if result.ok else "red"
        summary = (result.summary or "")[:SUMMARY_PREVIEW]
        suffix = "（需确认）" if result.requires_user_action else ""
        return f"[{color}]{mark}[/] [bold]{event.tool_name}[/]{suffix}: {summary}"
    if event.kind == "error" and event.content:
        return f"[red]❌ {event.content}[/]"
    if event.kind == "done":
        return "── 完成 ──"
    return ""


def render_user_message(message: str) -> str:
    """渲染用户输入行。"""
    return f"[bold cyan]你:[/] {message}"
