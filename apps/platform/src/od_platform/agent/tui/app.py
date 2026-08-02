"""ODPlatform Agent TUI（textual）。

四面板布局：
- 左上 Sessions：历史会话列表
- 右上 Conversation：对话消息流（用户/助手/工具事件）
- 左下 Tool Runs：工具执行记录
- 右下 Artifacts：最近运行产物
- 底部 Input：自然语言输入（/help 查看斜杠命令）

依赖 textual（软依赖）：未安装时 ``odp-agent --tui`` 回退交互 REPL。

@FileName:   app.py
@Function:   textual App：布局、事件渲染、斜杠命令、会话管理
"""

from __future__ import annotations

import logging

from od_platform.agent.client import OpenAIClient
from od_platform.agent.orchestrator import AgentConfig, AgentOrchestrator
from od_platform.agent.session import AgentSession, list_sessions
from od_platform.agent.tools import ToolRegistry
from od_platform.agent.tui.render import render_event, render_user_message

logger = logging.getLogger(__name__)

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, Input, RichLog, Static
except ImportError as exc:  # pragma: no cover - 软依赖降级
    logger.warning("textual 未安装，TUI 不可用: %s", exc)
    raise


class ODPAgentApp(App[None]):
    """ODPlatform Agent TUI 主应用。"""

    TITLE = "ODPlatform Agent"
    SUB_TITLE = "目标检测开发平台 AI 助手"

    CSS = """
    #left {
        width: 38%;
        border-right: solid #1f2f4d;
    }
    #right {
        width: 62%;
    }
    #session-list, #artifacts {
        height: 1fr;
        border: solid #1f2f4d;
        margin: 0 1;
        padding: 0 1;
    }
    #conversation, #tool-runs {
        border: solid #1f2f4d;
        margin: 0 1;
        padding: 0 1;
    }
    #conversation {
        height: 3fr;
    }
    #tool-runs {
        height: 1fr;
    }
    .panel-title {
        color: #7fb2f5;
        text-style: bold;
        margin: 1 1 0 1;
    }
    #prompt {
        margin: 1 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "退出"),
        Binding("ctrl+l", "clear_conversation", "清空对话"),
    ]

    def __init__(
        self,
        *,
        client: OpenAIClient,
        registry: ToolRegistry,
        config: AgentConfig,
        session: AgentSession | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.registry = registry
        self.config = config
        self.session = session
        if self.session is None:
            self.session = AgentSession.create(orchestrator_system_prompt())
        self.orchestrator = AgentOrchestrator(
            client=client,
            registry=registry,
            config=config,
            session=self.session,
        )

    # ---- 布局 ----

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left"):
                yield Static("会话", classes="panel-title")
                yield Static("", id="session-list")
                yield Static("工具运行", classes="panel-title")
                yield RichLog(id="tool-runs", highlight=True, markup=True, wrap=True)
            with Vertical(id="right"):
                yield Static("对话", classes="panel-title")
                yield RichLog(id="conversation", highlight=True, markup=True, wrap=True)
                yield Static("产物", classes="panel-title")
                yield Static("", id="artifacts")
        yield Input(id="prompt", placeholder="输入自然语言命令（/help 查看；Enter 发送）")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_sessions()
        self._refresh_artifacts()
        self.query_one("#prompt", Input).focus()
        conversation = self.query_one("#conversation", RichLog)
        conversation.write(f"[dim]会话 {self.session.session_id if self.session else '-'}[/]")

    # ---- 事件处理 ----

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        self.query_one("#prompt", Input).value = ""
        if not message:
            return
        if message.startswith("/"):
            self._handle_slash(message)
            return
        self.run_worker(self._agent_task(message), thread=True, exclusive=True)

    def action_clear_conversation(self) -> None:
        self.query_one("#conversation", RichLog).clear()
        if self.session is not None:
            self.session.clear_history()
            self.session.save()

    # ---- Agent 执行 ----

    def _agent_task(self, message: str):
        """构造后台线程执行函数（非生成器）。

        ``run_worker(thread=True)`` 不接受同步生成器（会退化为在 app
        事件循环内迭代，导致 ``call_from_thread`` 同线程报错），因此
        返回普通函数供 Worker 在线程中执行，事件经 ``call_from_thread``
        安全回传 UI 线程渲染。
        """
        conversation = self.query_one("#conversation", RichLog)
        tool_runs = self.query_one("#tool-runs", RichLog)

        def _task() -> None:
            self.call_from_thread(conversation.write, render_user_message(message))
            for event in self.orchestrator.run_stream(message):
                line = render_event(event)
                if not line:
                    continue
                self.call_from_thread(conversation.write, line)
                if event.kind in ("tool_start", "tool_result"):
                    self.call_from_thread(tool_runs.write, line)
                if event.kind == "done":
                    self.call_from_thread(self._refresh_artifacts)

        return _task

    # ---- 斜杠命令 ----

    def _handle_slash(self, command: str) -> None:
        conversation = self.query_one("#conversation", RichLog)
        cmd, _, arg = command.partition(" ")
        arg = arg.strip()

        if cmd == "/help":
            conversation.write("[dim]命令: /help /sessions /resume /open <id> /new /clear /artifacts /exit[/]")
        elif cmd == "/sessions":
            sessions = list_sessions(limit=20)
            if not sessions:
                conversation.write("[dim]暂无历史会话[/]")
            for session in sessions:
                marker = " ← 当前" if self.session is not None and session.session_id == self.session.session_id else ""
                conversation.write(f"  {session.session_id}  {session.title}{marker}")
        elif cmd == "/resume":
            from od_platform.agent.session import latest_session

            session = latest_session()
            if session is None:
                conversation.write("[dim]没有可恢复的会话[/]")
            else:
                self._switch_session(session)
        elif cmd == "/open":
            if not arg:
                conversation.write("[dim]用法: /open <session_id>[/]")
            else:
                session = AgentSession.load(arg)
                if session is None:
                    conversation.write(f"[red]会话 {arg} 不存在[/]")
                else:
                    self._switch_session(session)
        elif cmd == "/new":
            self._switch_session(AgentSession.create(orchestrator_system_prompt()))
        elif cmd == "/clear":
            self.action_clear_conversation()
        elif cmd == "/artifacts":
            self._refresh_artifacts()
        elif cmd == "/exit":
            self.exit()
        else:
            conversation.write(f"[dim]未知命令: {cmd}（/help 查看）[/]")

    def _switch_session(self, session: AgentSession) -> None:
        self.session = session
        self.orchestrator.session = session
        conversation = self.query_one("#conversation", RichLog)
        conversation.write(f"[dim]已切换会话: {session.session_id} ({session.title})[/]")
        self._refresh_sessions()

    # ---- 面板刷新 ----

    def _refresh_sessions(self) -> None:
        widget = self.query_one("#session-list", Static)
        sessions = list_sessions(limit=10)
        if not sessions:
            widget.update("  (无历史会话)")
            return
        lines = []
        for session in sessions:
            marker = "●" if self.session is not None and session.session_id == self.session.session_id else "○"
            lines.append(f"  {marker} {session.session_id}  {session.title}")
        widget.update("\n".join(lines))

    def _refresh_artifacts(self) -> None:
        widget = self.query_one("#artifacts", Static)
        try:
            result = self.registry.execute("list_run_artifacts", {"limit": 5})
        except Exception:  # noqa: BLE001 - 产物面板不阻断
            widget.update("  (产物读取失败)")
            return
        widget.update("  " + "\n  ".join((result.summary or "").splitlines()))


def orchestrator_system_prompt() -> str:
    """返回默认系统提示词（供会话创建）。"""
    from od_platform.agent.orchestrator import SYSTEM_PROMPT

    return SYSTEM_PROMPT


def run_tui(
    *,
    client: OpenAIClient,
    registry: ToolRegistry,
    config: AgentConfig,
    session: AgentSession | None = None,
) -> int:
    """启动 TUI（阻塞直至退出）。"""
    app = ODPAgentApp(client=client, registry=registry, config=config, session=session)
    app.run()
    return 0
