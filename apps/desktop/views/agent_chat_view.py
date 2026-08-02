"""AI 助手工作台视图：会话列表 + 对话消息流 + 配置 + 输入。

与 AgentController 协作：视图负责 UI 与事件渲染，控制器负责线程与会话。
"""

from __future__ import annotations

from controllers.agent_controller import AgentController
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from services.session_store_adapter import SessionStoreAdapter


class AgentChatView(QWidget):
    """AI 助手工作台页面。

    Signals:
        config_changed: (base_url, model, api_key) 配置字段变更（供外部同步）。
        status_changed: (状态, 详情) 供主窗口状态栏显示。
    """

    config_changed = Signal(str, str, str)
    status_changed = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = AgentController(store=SessionStoreAdapter())
        self._controller.event_ready.connect(self._on_agent_event)
        self._controller.completed.connect(self._on_completed)
        self._controller.cancelled.connect(self._on_cancelled)
        self._controller.failed.connect(self._on_failed)
        self._controller.session_ready.connect(self._on_session_ready)
        self._controller.status_changed.connect(self._on_status_changed)
        self._controller.busy_changed.connect(self._on_busy_changed)

        self._build_ui()
        self._connect_signals()
        self._refresh_session_list()

    # ---- UI 构建 ----

    def _build_ui(self) -> None:
        title_label = QLabel("AI 助手")
        title_label.setObjectName("SectionTitle")
        hint_label = QLabel("自然语言驱动平台任务；左侧可切换历史会话（多轮记忆）。")
        hint_label.setObjectName("HintText")

        # 配置区
        config_group = QGroupBox("API 配置")
        config_form = QFormLayout(config_group)
        self.base_url_edit = QLineEdit("")
        self.base_url_edit.setPlaceholderText("OpenAI 兼容 API 地址（必填），如 https://api.deepseek.com/v1")
        self.model_edit = QLineEdit("")
        self.model_edit.setPlaceholderText("模型名（必填），如 deepseek-chat")
        self.api_key_edit = QLineEdit("")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("API 密钥；留空读环境变量 OPENAI_API_KEY")
        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setRange(1, 100)
        self.max_iterations_spin.setValue(8)
        self.dry_run_check = QCheckBox("安全模式：训练/推理以计划模式执行")
        self.dry_run_check.setChecked(True)
        config_form.addRow("API 地址", self.base_url_edit)
        config_form.addRow("模型名", self.model_edit)
        config_form.addRow("API 密钥", self.api_key_edit)
        config_row = QWidget()
        config_row_layout = QHBoxLayout(config_row)
        config_row_layout.setContentsMargins(0, 0, 0, 0)
        config_row_layout.addWidget(self.max_iterations_spin)
        config_row_layout.addWidget(self.dry_run_check)
        config_form.addRow("迭代上限", config_row)

        # 会话列表
        session_group = QGroupBox("会话")
        session_layout = QVBoxLayout(session_group)
        session_buttons = QHBoxLayout()
        self.new_session_button = QPushButton("新建")
        self.resume_button = QPushButton("恢复最近")
        self.refresh_sessions_button = QPushButton("刷新")
        session_buttons.addWidget(self.new_session_button)
        session_buttons.addWidget(self.resume_button)
        session_buttons.addWidget(self.refresh_sessions_button)
        self.session_list = QListWidget()
        session_layout.addLayout(session_buttons)
        session_layout.addWidget(self.session_list)

        # 消息流
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        # 输入区
        self.input = QTextEdit()
        self.input.setPlaceholderText("用自然语言描述目标检测任务，如：用最快的模型训练 rsod 数据集")
        self.input.setMaximumHeight(90)
        self.send_button = QPushButton("发送")
        self.send_button.setProperty("primary", True)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        controls = QHBoxLayout()
        controls.addWidget(self.send_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)

        # 布局
        left = QVBoxLayout()
        left.addWidget(title_label)
        left.addWidget(hint_label)
        left.addWidget(config_group)
        left.addWidget(session_group)

        right = QVBoxLayout()
        output_title = QLabel("对话输出")
        output_title.setObjectName("SectionTitle")
        right.addWidget(output_title)
        right.addWidget(self.output)
        right.addWidget(self.input)
        right.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_container = QWidget()
        left_container.setLayout(left)
        right_container = QWidget()
        right_container.setLayout(right)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([340, 900])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

    def _connect_signals(self) -> None:
        self.send_button.clicked.connect(self._submit)
        self.stop_button.clicked.connect(self._controller.stop)
        self.new_session_button.clicked.connect(self._new_session)
        self.resume_button.clicked.connect(self._resume_latest)
        self.refresh_sessions_button.clicked.connect(self._refresh_session_list)
        self.session_list.itemDoubleClicked.connect(self._open_session)
        self.base_url_edit.textChanged.connect(self._emit_config_changed)
        self.model_edit.textChanged.connect(self._emit_config_changed)
        self.api_key_edit.textChanged.connect(self._emit_config_changed)

    # ---- 交互 ----

    def _submit(self) -> None:
        message = self.input.toPlainText().strip()
        base_url = self.base_url_edit.text().strip()
        model = self.model_edit.text().strip()
        api_key = self.api_key_edit.text().strip() or None
        if not message:
            self._append("⚠ 请输入任务描述")
            return
        if not base_url or not model:
            self._append("⚠ 请填写 API 地址与模型名")
            return
        self._append(f"> {message}")
        self.input.clear()
        self._controller.submit(
            message,
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_iterations=self.max_iterations_spin.value(),
            dry_run=self.dry_run_check.isChecked(),
        )

    def _new_session(self) -> None:
        self._controller.new_session()
        self._append("── 已新建会话 ──")
        self._refresh_session_list()

    def _resume_latest(self) -> None:
        session = self._controller.resume_latest()
        if session is None:
            self._append("没有可恢复的会话")
        else:
            self._append(f"── 已恢复会话 {session.session_id} ──")
            self._refresh_session_list()

    def _open_session(self, item) -> None:
        session_id = item.data(Qt.ItemDataRole.UserRole)
        session = self._controller.load_session(session_id)
        if session is not None:
            self._append(f"── 已切换会话 {session.session_id} ──")
            self._refresh_session_list()

    def _refresh_session_list(self) -> None:
        self.session_list.clear()
        sessions = self._controller.list_sessions(limit=20)
        current_id = getattr(self._controller.session, "session_id", None) if self._controller.session else None
        for session in sessions:
            marker = "● " if session.session_id == current_id else "○ "
            self.session_list.addItem(f"{marker}{session.session_id}  {session.title}")
            self.session_list.item(self.session_list.count() - 1).setData(
                Qt.ItemDataRole.UserRole, session.session_id
            )

    def _emit_config_changed(self, _text: str = "") -> None:
        self.config_changed.emit(
            self.base_url_edit.text(),
            self.model_edit.text(),
            self.api_key_edit.text(),
        )

    # ---- 外部接口 ----

    def set_config(self, base_url: str, model: str, api_key: str) -> None:
        """外部（任务页）同步配置字段。"""
        self.base_url_edit.setText(base_url)
        self.model_edit.setText(model)
        self.api_key_edit.setText(api_key)

    def config_values(self) -> tuple[str, str, str]:
        return (
            self.base_url_edit.text(),
            self.model_edit.text(),
            self.api_key_edit.text(),
        )

    # ---- 事件渲染 ----

    def _append(self, text: str) -> None:
        self.output.append(text)

    def _on_agent_event(self, event: object) -> None:
        kind = getattr(event, "kind", "")
        content = getattr(event, "content", None)
        if kind == "message" and content:
            self._append(f"🤖 {content}")
        elif kind == "tool_start":
            self._append(f"→ 调用工具 {event.tool_name}({content or ''})")
        elif kind == "tool_result" and event.tool_result is not None:
            status = "成功" if event.tool_result.ok else "失败"
            summary = (event.tool_result.summary or "")[:200]
            suffix = "（需确认）" if getattr(event, "requires_user_action", False) else ""
            self._append(f"← {event.tool_name} {status}{suffix}: {summary}")
        elif kind == "error":
            self._append(f"❌ 错误: {content}")
        elif kind == "done":
            self._append("── 完成 ──")

    def _on_completed(self, final_text: str) -> None:
        if final_text:
            self._append(f"🤖 {final_text}")

    def _on_cancelled(self) -> None:
        self._append("⏹ 已取消")

    def _on_failed(self, message: str) -> None:
        self._append(f"❌ 失败: {message}")

    def _on_session_ready(self, _session: object) -> None:
        self._refresh_session_list()

    def _on_status_changed(self, state: str, detail: str) -> None:
        # 转发给主窗口状态栏（内嵌 widget 不修改顶层窗口标题）
        self.status_changed.emit(state, detail)

    def _on_busy_changed(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
