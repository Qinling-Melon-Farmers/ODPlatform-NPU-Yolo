"""ODPlatform Agent TUI 子系统（textual 软依赖）。

未安装 textual 时调用 ``run_tui`` 抛 ImportError，由 CLI 层捕获降级。
"""

from __future__ import annotations

from od_platform.agent.tui.render import render_event, render_user_message

__all__ = ["render_event", "render_user_message"]
