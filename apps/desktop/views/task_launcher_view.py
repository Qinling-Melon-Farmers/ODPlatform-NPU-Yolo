"""任务启动页视图：结构化表单 + 命令预览 + CLI 任务执行。

与 TaskController 协作：视图负责 UI/参数收集/命令预览（含 API Key 脱敏），
控制器负责线程。跨页数据（AI 配置同步、模型族、模型回填）经信号/方法。
"""

from __future__ import annotations

import sys

import task_builder
from controllers.task_controller import TaskController
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from services import paths as desktop_paths
from services.file_utils import split_extra_args

from views.ui_helpers import form_page, with_buttons


class TaskLauncherView(QWidget):
    """任务启动页（11 类 CLI 任务的结构化启动）。

    Signals:
        status_changed:       (状态, 详情) 主窗口状态栏。
        task_finished:        任务退出码（主窗口刷新结果页）。
        agent_config_changed: (base_url, model, api_key) AI 配置变更（AI 桥接）。
    """

    status_changed = Signal(str, str)
    task_finished = Signal(int)
    agent_config_changed = Signal(str, str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = TaskController(
            cwd=str(desktop_paths.ROOT_DIR),
            platform_src=str(desktop_paths.PLATFORM_SRC),
        )
        self._controller.output_ready.connect(self._on_output_ready)
        self._controller.finished.connect(self._on_finished)
        self._controller.failed.connect(self._on_failed)
        self._controller.busy_changed.connect(self._on_busy_changed)

        self._create_widgets()
        self._build_ui()
        self._connect_signals()
        self._switch_task_page(self.desktop_task_combo.currentIndex())
        self._refresh_task_preview()

    # ---- 控件创建（原 main.py __init__ 任务区） ----

    def _create_widgets(self) -> None:
        self.desktop_task_combo = QComboBox()
        self.desktop_task_combo.addItems(list(task_builder.TASK_NAMES))
        self.task_stack = QStackedWidget()

        self.import_dataset_edit = QLineEdit("steel-surface-defect")
        self.import_zip_edit = QLineEdit("")
        self.import_zip_edit.setPlaceholderText("VOC/YOLO zip 路径")
        self.import_format_combo = QComboBox()
        self.import_format_combo.addItems(["voc", "yolo"])
        self.import_overwrite_check = QCheckBox("允许覆盖已有 raw 数据目录")
        self.import_extra_args_edit = QLineEdit("")
        self.import_extra_args_edit.setPlaceholderText("追加 CLI 参数")

        self.transform_dataset_edit = QLineEdit("steel-surface-defect")
        self.transform_format_combo = QComboBox()
        self.transform_format_combo.addItems(["pascal_voc", "coco", "yolo"])
        self.transform_task_combo = QComboBox()
        self.transform_task_combo.addItems(["detect"])
        self.transform_extra_args_edit = QLineEdit("")
        self.transform_extra_args_edit.setPlaceholderText("例如 --split-strategy random --seed 1210")

        self.validate_dataset_edit = QLineEdit("steel-surface-defect")
        self.validate_task_combo = QComboBox()
        self.validate_task_combo.addItems(["detect"])
        self.validate_executor_edit = QLineEdit("")
        self.validate_extra_args_edit = QLineEdit("")
        self.validate_extra_args_edit.setPlaceholderText("例如 --verbose")

        self.eval_model_edit = QLineEdit("")
        self.eval_dataset_edit = QLineEdit("steel-surface-defect")
        self.eval_config_edit = QLineEdit("val")
        self.eval_device_edit = QLineEdit("0")
        self.eval_executor_edit = QLineEdit("")
        self.eval_name_edit = QLineEdit("desktop-eval")
        self.eval_extra_args_edit = QLineEdit("")
        self.eval_extra_args_edit.setPlaceholderText("追加 CLI 参数")

        self.train_model_edit = QLineEdit(str(desktop_paths.default_model()))
        self.train_dataset_edit = QLineEdit("steel-surface-defect")
        self.train_config_edit = QLineEdit("train")
        self.train_device_edit = QLineEdit("0")
        self.train_executor_edit = QLineEdit("")
        self.train_name_edit = QLineEdit("desktop-train")
        self.train_epochs_spin = QSpinBox()
        self.train_epochs_spin.setRange(1, 10000)
        self.train_epochs_spin.setValue(100)
        self.train_batch_spin = QSpinBox()
        self.train_batch_spin.setRange(1, 4096)
        self.train_batch_spin.setValue(16)
        self.train_workers_spin = QSpinBox()
        self.train_workers_spin.setRange(0, 128)
        self.train_workers_spin.setValue(4)
        self.train_dry_run_check = QCheckBox("dry-run：只生成计划和日志，不启动训练")
        self.train_dry_run_check.setChecked(True)
        self.train_extra_args_edit = QLineEdit("")
        self.train_extra_args_edit.setPlaceholderText("例如 --imgsz 640 --no-archive")

        self.reset_dry_run_check = QCheckBox("dry-run：只预览，不删除")
        self.reset_dry_run_check.setChecked(True)
        self.reset_yes_check = QCheckBox("确认执行 --yes")
        self.reset_force_check = QCheckBox("跳过交互确认 --force")
        self.reset_backup_check = QCheckBox("删除前备份 --backup")
        self.reset_extra_args_edit = QLineEdit("")
        self.reset_extra_args_edit.setPlaceholderText("追加 CLI 参数")

        self.plot_csv_edit = QLineEdit(str(desktop_paths.default_results_csv()))
        self.plot_output_edit = QLineEdit("")
        self.plot_output_edit.setPlaceholderText("输出 PNG 路径；留空则使用默认输出")
        self.plot_summary_edit = QLineEdit("")
        self.plot_summary_edit.setPlaceholderText("可选 summary JSON 路径")
        self.plot_matplotx_check = QCheckBox("启用 matplotx 风格")
        self.plot_extra_args_edit = QLineEdit("")
        self.plot_extra_args_edit.setPlaceholderText("追加 CLI 参数")

        self.list_models_family_combo = QComboBox()
        self.list_models_recommend_edit = QLineEdit("")
        self.list_models_recommend_edit.setPlaceholderText("自然语言推荐，如：最快 / 最准")
        self.list_models_limit_spin = QSpinBox()
        self.list_models_limit_spin.setRange(1, 100)
        self.list_models_limit_spin.setValue(10)
        self.list_models_json_check = QCheckBox("JSON 格式输出")
        self.list_models_extra_args_edit = QLineEdit("")
        self.list_models_extra_args_edit.setPlaceholderText("追加 CLI 参数")

        self.annotate_dataset_edit = QLineEdit("")
        self.annotate_dataset_edit.setPlaceholderText("数据集名称，位于 data/raw/<name>/")
        self.annotate_classes_edit = QLineEdit("")
        self.annotate_classes_edit.setPlaceholderText("空格分隔的类别名，如：cat dog ship")
        self.annotate_resume_check = QCheckBox("恢复中断的标注会话（跳过已标注图片）")
        self.annotate_extra_args_edit = QLineEdit("")
        self.annotate_extra_args_edit.setPlaceholderText("追加 CLI 参数")

        self.auto_annotate_dataset_edit = QLineEdit("")
        self.auto_annotate_dataset_edit.setPlaceholderText("数据集名称，位于 data/raw/<name>/")
        self.auto_annotate_classes_edit = QLineEdit("")
        self.auto_annotate_classes_edit.setPlaceholderText("空格分隔的类别名，如：cat dog ship")
        self.auto_annotate_prompt_edit = QLineEdit("框出所有目标")
        self.auto_annotate_prompt_edit.setPlaceholderText("自然语言标注指令，如：框出所有飞机")
        self.auto_annotate_base_url_edit = QLineEdit("")
        self.auto_annotate_base_url_edit.setPlaceholderText("OpenAI 兼容 API 地址（必填），如 DashScope/Qwen 兼容端点")
        self.auto_annotate_model_edit = QLineEdit("")
        self.auto_annotate_model_edit.setPlaceholderText("视觉模型名（必填），如 qwen-vl-max / glm-4.5v-turbo")
        self.auto_annotate_api_key_edit = QLineEdit("")
        self.auto_annotate_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.auto_annotate_api_key_edit.setPlaceholderText("API 密钥；留空读环境变量 OPENAI_API_KEY")
        self.auto_annotate_limit_spin = QSpinBox()
        self.auto_annotate_limit_spin.setRange(0, 1_000_000)
        self.auto_annotate_limit_spin.setSpecialValueText("不限")
        self.auto_annotate_dry_run_check = QCheckBox("dry-run：只统计待标注数量，不调用 API")
        self.auto_annotate_dry_run_check.setChecked(True)
        self.auto_annotate_extra_args_edit = QLineEdit("")
        self.auto_annotate_extra_args_edit.setPlaceholderText("追加 CLI 参数")

        self.ai_task_prompt_edit = QLineEdit("")
        self.ai_task_prompt_edit.setPlaceholderText("自然语言任务，如：用最快的模型训练 rsod 数据集")
        self.ai_task_base_url_edit = QLineEdit("")
        self.ai_task_base_url_edit.setPlaceholderText("OpenAI 兼容 API 地址（必填），如 https://api.deepseek.com/v1")
        self.ai_task_model_edit = QLineEdit("")
        self.ai_task_model_edit.setPlaceholderText("模型名（必填），如 deepseek-chat")
        self.ai_task_api_key_edit = QLineEdit("")
        self.ai_task_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.ai_task_api_key_edit.setPlaceholderText("API 密钥；留空读环境变量 OPENAI_API_KEY")
        self.ai_task_extra_args_edit = QLineEdit("")
        self.ai_task_extra_args_edit.setPlaceholderText("追加 CLI 参数")

        self.task_command_preview = QTextEdit()
        self.task_command_preview.setReadOnly(True)
        self.task_command_preview.setMaximumHeight(100)
        self.task_output = QTextEdit()
        self.task_output.setReadOnly(True)
        self.task_start_button = QPushButton("启动任务")
        self.task_start_button.setProperty("primary", True)
        self.task_stop_button = QPushButton("停止任务")
        self.task_stop_button.setEnabled(False)

    # ---- UI 构建（原 _build_tasks_tab） ----

    def _build_ui(self) -> None:
        task_group = QGroupBox("选择任务")
        task_form = QFormLayout(task_group)
        task_form.addRow("任务类型", self.desktop_task_combo)

        self.task_stack.addWidget(
            form_page(
                "导入数据集参数",
                [
                    ("数据集名称", self.import_dataset_edit),
                    ("数据 zip", with_buttons(self.import_zip_edit, [("选择 zip", self._browse_import_zip)])),
                    ("标注格式", self.import_format_combo),
                    ("", self.import_overwrite_check),
                    ("追加参数", self.import_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            form_page(
                "数据转换参数",
                [
                    ("数据集名称", self.transform_dataset_edit),
                    ("标注格式", self.transform_format_combo),
                    ("任务类型", self.transform_task_combo),
                    ("追加参数", self.transform_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            form_page(
                "数据质检参数",
                [
                    ("数据集名称", self.validate_dataset_edit),
                    ("任务类型", self.validate_task_combo),
                    ("执行人", self.validate_executor_edit),
                    ("追加参数", self.validate_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            form_page(
                "模型评估参数",
                [
                    ("模型权重", with_buttons(self.eval_model_edit, [("选择权重", self._browse_eval_model)])),
                    ("数据集名称", self.eval_dataset_edit),
                    ("评估配置", self.eval_config_edit),
                    ("设备", self.eval_device_edit),
                    ("执行人", self.eval_executor_edit),
                    ("运行名", self.eval_name_edit),
                    ("追加参数", self.eval_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            form_page(
                "模型训练参数",
                [
                    ("模型权重/名称", with_buttons(self.train_model_edit, [("选择权重", self._browse_train_model)])),
                    ("数据集名称", self.train_dataset_edit),
                    ("训练配置", self.train_config_edit),
                    ("设备", self.train_device_edit),
                    ("执行人", self.train_executor_edit),
                    ("运行名", self.train_name_edit),
                    ("epochs", self.train_epochs_spin),
                    ("batch", self.train_batch_spin),
                    ("workers", self.train_workers_spin),
                    ("", self.train_dry_run_check),
                    ("追加参数", self.train_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            form_page(
                "项目重置参数",
                [
                    ("", self.reset_dry_run_check),
                    ("", self.reset_backup_check),
                    ("", self.reset_yes_check),
                    ("", self.reset_force_check),
                    ("追加参数", self.reset_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            form_page(
                "训练曲线生成参数",
                [
                    ("results.csv", with_buttons(self.plot_csv_edit, [("选择 CSV", self._browse_plot_csv)])),
                    ("输出图片", with_buttons(self.plot_output_edit, [("选择 PNG", self._browse_plot_output)])),
                    ("摘要 JSON", with_buttons(self.plot_summary_edit, [("选择 JSON", self._browse_plot_summary)])),
                    ("", self.plot_matplotx_check),
                    ("追加参数", self.plot_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            form_page(
                "列出模型参数",
                [
                    ("模型系列", self.list_models_family_combo),
                    ("推荐描述", self.list_models_recommend_edit),
                    ("数量上限", self.list_models_limit_spin),
                    ("", self.list_models_json_check),
                    ("追加参数", self.list_models_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            form_page(
                "数据标注参数",
                [
                    ("数据集名称", self.annotate_dataset_edit),
                    ("类别列表", self.annotate_classes_edit),
                    ("", self.annotate_resume_check),
                    ("追加参数", self.annotate_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            form_page(
                "自动标注参数（VLM）",
                [
                    ("数据集名称", self.auto_annotate_dataset_edit),
                    ("类别列表", self.auto_annotate_classes_edit),
                    ("标注指令", self.auto_annotate_prompt_edit),
                    ("API 地址", self.auto_annotate_base_url_edit),
                    ("视觉模型", self.auto_annotate_model_edit),
                    ("API 密钥", self.auto_annotate_api_key_edit),
                    ("数量上限", self.auto_annotate_limit_spin),
                    ("", self.auto_annotate_dry_run_check),
                    ("追加参数", self.auto_annotate_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            form_page(
                "AI 任务参数",
                [
                    ("自然语言任务", self.ai_task_prompt_edit),
                    ("API 地址", self.ai_task_base_url_edit),
                    ("模型名", self.ai_task_model_edit),
                    ("API 密钥", self.ai_task_api_key_edit),
                    ("追加参数", self.ai_task_extra_args_edit),
                ],
            )
        )

        controls = QHBoxLayout()
        controls.addWidget(self.task_start_button)
        controls.addWidget(self.task_stop_button)
        controls.addStretch(1)

        left = QVBoxLayout()
        title = QLabel("任务启动")
        title.setObjectName("SectionTitle")
        hint = QLabel("从桌面端调用现有 CLI。训练默认 dry-run，避免误触发长时间任务。")
        hint.setObjectName("HintText")
        left.addWidget(title)
        left.addWidget(hint)
        left.addWidget(task_group)
        left.addWidget(self.task_stack)
        left.addLayout(controls)
        left.addWidget(QLabel("命令预览"))
        left.addWidget(self.task_command_preview)
        left.addStretch(1)

        right = QVBoxLayout()
        output_title = QLabel("任务输出")
        output_title.setObjectName("SectionTitle")
        right.addWidget(output_title)
        right.addWidget(self.task_output)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_container = QWidget()
        left_container.setLayout(left)
        right_container = QWidget()
        right_container.setLayout(right)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([480, 780])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

    # ---- 信号连接（原 _connect_signals 任务区） ----

    def _connect_signals(self) -> None:
        self.desktop_task_combo.currentIndexChanged.connect(self._switch_task_page)
        self.desktop_task_combo.currentTextChanged.connect(lambda _text: self._refresh_task_preview())
        for line_edit in (
            self.import_dataset_edit,
            self.import_zip_edit,
            self.import_extra_args_edit,
            self.transform_dataset_edit,
            self.transform_extra_args_edit,
            self.validate_dataset_edit,
            self.validate_executor_edit,
            self.validate_extra_args_edit,
            self.eval_model_edit,
            self.eval_dataset_edit,
            self.eval_config_edit,
            self.eval_device_edit,
            self.eval_executor_edit,
            self.eval_name_edit,
            self.eval_extra_args_edit,
            self.train_model_edit,
            self.train_dataset_edit,
            self.train_config_edit,
            self.train_device_edit,
            self.train_executor_edit,
            self.train_name_edit,
            self.train_extra_args_edit,
            self.reset_extra_args_edit,
            self.plot_csv_edit,
            self.plot_output_edit,
            self.plot_summary_edit,
            self.plot_extra_args_edit,
            self.list_models_recommend_edit,
            self.list_models_extra_args_edit,
            self.annotate_dataset_edit,
            self.annotate_classes_edit,
            self.annotate_extra_args_edit,
            self.auto_annotate_dataset_edit,
            self.auto_annotate_classes_edit,
            self.auto_annotate_prompt_edit,
            self.auto_annotate_base_url_edit,
            self.auto_annotate_model_edit,
            self.auto_annotate_api_key_edit,
            self.auto_annotate_extra_args_edit,
            self.ai_task_prompt_edit,
            self.ai_task_base_url_edit,
            self.ai_task_model_edit,
            self.ai_task_api_key_edit,
            self.ai_task_extra_args_edit,
        ):
            line_edit.textChanged.connect(lambda _text: self._refresh_task_preview())
        for combo_box in (
            self.import_format_combo,
            self.transform_format_combo,
            self.transform_task_combo,
            self.validate_task_combo,
            self.list_models_family_combo,
        ):
            combo_box.currentTextChanged.connect(lambda _text: self._refresh_task_preview())
        for spin_box in (
            self.train_epochs_spin,
            self.train_batch_spin,
            self.train_workers_spin,
            self.list_models_limit_spin,
            self.auto_annotate_limit_spin,
        ):
            spin_box.valueChanged.connect(lambda _value: self._refresh_task_preview())
        self.import_overwrite_check.stateChanged.connect(lambda _value: self._refresh_task_preview())
        self.train_dry_run_check.stateChanged.connect(lambda _value: self._refresh_task_preview())
        self.list_models_json_check.stateChanged.connect(lambda _value: self._refresh_task_preview())
        self.annotate_resume_check.stateChanged.connect(lambda _value: self._refresh_task_preview())
        self.auto_annotate_dry_run_check.stateChanged.connect(lambda _value: self._refresh_task_preview())
        for check_box in (
            self.reset_dry_run_check,
            self.reset_yes_check,
            self.reset_force_check,
            self.reset_backup_check,
            self.plot_matplotx_check,
        ):
            check_box.stateChanged.connect(lambda _value: self._refresh_task_preview())

        # AI 配置变更信号（AI 桥接）
        for edit in (self.ai_task_base_url_edit, self.ai_task_model_edit, self.ai_task_api_key_edit):
            edit.textChanged.connect(self._emit_agent_config_changed)

        self.task_start_button.clicked.connect(self._on_start_clicked)
        self.task_stop_button.clicked.connect(self._on_stop_clicked)

    # ---- 交互 ----

    @Slot()
    def _on_start_clicked(self) -> None:
        try:
            module, args = self._build_task_command()
        except ValueError as exc:
            self.task_output.append(f"参数错误: {exc}")
            self.status_changed.emit("失败", str(exc))
            return
        self.task_output.clear()
        self.task_output.append("启动桌面端任务")
        self._controller.start(module, args)

    @Slot()
    def _on_stop_clicked(self) -> None:
        self._controller.stop()
        self.status_changed.emit("停止中", "正在停止桌面端任务")

    @Slot(str)
    def _on_output_ready(self, line: str) -> None:
        self.task_output.append(line)

    @Slot(int)
    def _on_finished(self, exit_code: int) -> None:
        self.task_output.append(f"任务结束，退出码: {exit_code}")
        if exit_code == 0:
            self.status_changed.emit("完成", "桌面端任务完成")
        else:
            self.status_changed.emit("失败", f"桌面端任务退出码 {exit_code}")
        self.task_finished.emit(exit_code)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.task_output.append(f"任务失败: {message}")
        self.status_changed.emit("失败", message)

    @Slot(bool)
    def _on_busy_changed(self, busy: bool) -> None:
        self.task_start_button.setEnabled(not busy)
        self.task_stop_button.setEnabled(busy)

    # ---- 命令构建 ----

    def _switch_task_page(self, index: int) -> None:
        self.task_stack.setCurrentIndex(max(0, index))

    def _build_task_command(self) -> tuple[str, list[str]]:
        """收集任务参数并委托 task_builder 构造 CLI 命令。"""
        return task_builder.build_task_command(
            self.desktop_task_combo.currentText(),
            self._collect_task_params(),
        )

    def _collect_task_params(self) -> dict:
        """按当前任务收集控件值 → task_builder 参数字典。"""
        task_name = self.desktop_task_combo.currentText()

        if task_name == "导入数据集":
            return {
                "zip_path": self.import_zip_edit.text().strip(),
                "dataset": self.import_dataset_edit.text().strip(),
                "format": self.import_format_combo.currentText(),
                "overwrite": self.import_overwrite_check.isChecked(),
                "extra_args": split_extra_args(self.import_extra_args_edit.text().strip()),
            }
        if task_name == "数据转换":
            return {
                "dataset": self.transform_dataset_edit.text().strip(),
                "format": self.transform_format_combo.currentText(),
                "task": self.transform_task_combo.currentText(),
                "extra_args": split_extra_args(self.transform_extra_args_edit.text().strip()),
            }
        if task_name == "数据质检":
            return {
                "dataset": self.validate_dataset_edit.text().strip(),
                "task": self.validate_task_combo.currentText(),
                "executor": self.validate_executor_edit.text().strip(),
                "extra_args": split_extra_args(self.validate_extra_args_edit.text().strip()),
            }
        if task_name == "模型评估":
            return {
                "model": self.eval_model_edit.text().strip(),
                "dataset": self.eval_dataset_edit.text().strip(),
                "config": self.eval_config_edit.text().strip(),
                "device": self.eval_device_edit.text().strip(),
                "executor": self.eval_executor_edit.text().strip(),
                "name": self.eval_name_edit.text().strip(),
                "extra_args": split_extra_args(self.eval_extra_args_edit.text().strip()),
            }
        if task_name == "模型训练":
            return {
                "model": self.train_model_edit.text().strip(),
                "dataset": self.train_dataset_edit.text().strip(),
                "config": self.train_config_edit.text().strip(),
                "device": self.train_device_edit.text().strip(),
                "executor": self.train_executor_edit.text().strip(),
                "name": self.train_name_edit.text().strip(),
                "epochs": self.train_epochs_spin.value(),
                "batch": self.train_batch_spin.value(),
                "workers": self.train_workers_spin.value(),
                "dry_run": self.train_dry_run_check.isChecked(),
                "extra_args": split_extra_args(self.train_extra_args_edit.text().strip()),
            }
        if task_name == "项目重置":
            return {
                "dry_run": self.reset_dry_run_check.isChecked(),
                "backup": self.reset_backup_check.isChecked(),
                "yes": self.reset_yes_check.isChecked(),
                "force": self.reset_force_check.isChecked(),
                "extra_args": split_extra_args(self.reset_extra_args_edit.text().strip()),
            }
        if task_name == "训练曲线生成":
            return {
                "csv_path": self.plot_csv_edit.text().strip(),
                "output": self.plot_output_edit.text().strip(),
                "summary": self.plot_summary_edit.text().strip(),
                "matplotx": self.plot_matplotx_check.isChecked(),
                "extra_args": split_extra_args(self.plot_extra_args_edit.text().strip()),
            }
        if task_name == "列出模型":
            family = self.list_models_family_combo.currentText()
            return {
                "family": None if family == "全部" else family,
                "recommend": self.list_models_recommend_edit.text().strip(),
                "limit": self.list_models_limit_spin.value(),
                "json": self.list_models_json_check.isChecked(),
                "extra_args": split_extra_args(self.list_models_extra_args_edit.text().strip()),
            }
        if task_name == "数据标注":
            return {
                "dataset": self.annotate_dataset_edit.text().strip(),
                "classes": self.annotate_classes_edit.text().split(),
                "resume": self.annotate_resume_check.isChecked(),
                "extra_args": split_extra_args(self.annotate_extra_args_edit.text().strip()),
            }
        if task_name == "自动标注":
            limit = self.auto_annotate_limit_spin.value()
            return {
                "dataset": self.auto_annotate_dataset_edit.text().strip(),
                "classes": self.auto_annotate_classes_edit.text().split(),
                "prompt": self.auto_annotate_prompt_edit.text().strip(),
                "base_url": self.auto_annotate_base_url_edit.text().strip(),
                "model": self.auto_annotate_model_edit.text().strip(),
                "api_key": self.auto_annotate_api_key_edit.text().strip(),
                "limit": limit or None,
                "dry_run": self.auto_annotate_dry_run_check.isChecked(),
                "extra_args": split_extra_args(self.auto_annotate_extra_args_edit.text().strip()),
            }
        if task_name == "AI 任务":
            return {
                "prompt": self.ai_task_prompt_edit.text().strip(),
                "base_url": self.ai_task_base_url_edit.text().strip(),
                "model": self.ai_task_model_edit.text().strip(),
                "api_key": self.ai_task_api_key_edit.text().strip(),
                "extra_args": split_extra_args(self.ai_task_extra_args_edit.text().strip()),
            }
        raise ValueError(f"未知任务: {task_name}")

    def _refresh_task_preview(self) -> None:
        try:
            module, args = self._build_task_command()
            command = " ".join([sys.executable, "-m", module, *args])
            # 密钥脱敏：避免命令预览/截图泄露 API Key
            for secret in (self.auto_annotate_api_key_edit.text(), self.ai_task_api_key_edit.text()):
                if secret:
                    command = command.replace(secret, "***")
        except ValueError as exc:
            command = f"参数待补全: {exc}"
        self.task_command_preview.setPlainText(command)

    # ---- 外部接口（MainWindow 唯一入口） ----

    def set_agent_config(self, base_url: str, model: str, api_key: str) -> None:
        """设置 AI 任务页 API 配置（AI 桥接，不发信号）。"""
        self.ai_task_base_url_edit.setText(base_url)
        self.ai_task_model_edit.setText(model)
        self.ai_task_api_key_edit.setText(api_key)

    def select_task(self, name: str) -> None:
        """切换任务（如模型目录一键应用）。"""
        self.desktop_task_combo.setCurrentText(name)

    def set_train_model(self, name: str) -> None:
        """回填训练模型字段（模型目录一键应用）。"""
        self.train_model_edit.setText(name)

    def set_model_families(self, families: list[str]) -> None:
        """填充列出模型系列下拉（model_catalog families_changed）。"""
        combo = self.list_models_family_combo
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(["全部", *families])
        combo.setCurrentText(current if current in ("全部", *families) else "全部")
        combo.blockSignals(False)
        self._refresh_task_preview()

    def shutdown(self) -> None:
        """窗口关闭前停止运行中的任务。"""
        self._controller.stop()

    # ---- 浏览回调 ----

    def _browse_import_zip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择数据集 zip", str(desktop_paths.ROOT_DIR.parent), "Zip files (*.zip);;All files (*)")
        if path:
            self.import_zip_edit.setText(path)

    def _browse_eval_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", str(desktop_paths.ROOT_DIR), "PyTorch weights (*.pt);;All files (*)")
        if path:
            self.eval_model_edit.setText(path)

    def _browse_train_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", str(desktop_paths.ROOT_DIR), "PyTorch weights (*.pt);;All files (*)")
        if path:
            self.train_model_edit.setText(path)

    def _browse_plot_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 results.csv", str(desktop_paths.ROOT_DIR / "runs"), "CSV files (*.csv);;All files (*)")
        if path:
            self.plot_csv_edit.setText(path)

    def _browse_plot_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "选择输出图片", str(desktop_paths.ROOT_DIR / "runs" / "training_summary.png"), "PNG files (*.png);;All files (*)")
        if path:
            self.plot_output_edit.setText(path)

    def _browse_plot_summary(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "选择摘要 JSON", str(desktop_paths.ROOT_DIR / "runs" / "training_summary.json"), "JSON files (*.json);;All files (*)")
        if path:
            self.plot_summary_edit.setText(path)

    def _emit_agent_config_changed(self, _text: str = "") -> None:
        self.agent_config_changed.emit(
            self.ai_task_base_url_edit.text(),
            self.ai_task_model_edit.text(),
            self.ai_task_api_key_edit.text(),
        )
