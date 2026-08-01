"""PySide6 desktop workbench for ODPlatform Studio."""

from __future__ import annotations

import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
PLATFORM_SRC = ROOT_DIR / "apps" / "platform" / "src"

try:
    import cv2
    from PySide6.QtCore import Qt, QThread, QUrl, Slot
    from PySide6.QtGui import QDesktopServices, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QPushButton,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - optional desktop dependencies.
    print(
        "桌面端需要 PySide6 和 opencv-python。请在 odplat 环境中安装后再运行：\n"
        "  conda activate odplat\n"
        "  pip install PySide6 opencv-python\n"
        f"当前缺失: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

from agent_worker import AgentWorker  # noqa: E402
from infer_worker import InferWorker  # noqa: E402
from task_worker import CommandWorker  # noqa: E402


class MainWindow(QMainWindow):
    """Local desktop workbench for inference and result inspection."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ODPlatform Studio")
        self.resize(1440, 900)
        self._thread: QThread | None = None
        self._worker: InferWorker | None = None
        self._task_thread: QThread | None = None
        self._task_worker: CommandWorker | None = None
        self._agent_thread: QThread | None = None
        self._agent_worker: AgentWorker | None = None
        self._last_output_dir: Path | None = None
        self._training_plots: list[Path] = []
        self._training_plot_index = 0

        self.model_edit = QLineEdit(str(_default_model()))
        self.source_edit = QLineEdit(str(_default_source()))
        self.runtime_edit = QLineEdit("")
        self.pipeline_edit = QLineEdit(str(ROOT_DIR / "apps" / "platform" / "configs" / "runtime" / "infer_pipeline.yaml"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect"])
        self.conf_edit = QLineEdit("0.25")
        self.iou_edit = QLineEdit("0.70")
        self.device_edit = QLineEdit("0")
        self.name_edit = QLineEdit("desktop-demo")
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 10000)
        self.max_det_spin.setValue(300)
        self.classes_edit = QLineEdit("")
        self.classes_edit.setPlaceholderText("如 0,2,5；空=全部类别")
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setRange(0, 1_000_000)
        self.max_frames_spin.setSpecialValueText("不限")
        self.vid_stride_spin = QSpinBox()
        self.vid_stride_spin.setRange(1, 10_000)
        self.vid_stride_spin.setValue(1)
        self.save_check = QCheckBox("保存推理结果")
        self.threaded_check = QCheckBox("多级流水线")
        self.threaded_check.setChecked(True)

        self.start_button = QPushButton("启动")
        self.start_button.setProperty("primary", True)
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.open_output_button = QPushButton("打开输出目录")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.open_output_button.setEnabled(False)

        self.image_label = QLabel("等待启动推理")
        self.image_label.setObjectName("PreviewPane")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(860, 560)
        self.status_label = QLabel("状态：就绪")
        self.status_label.setObjectName("StatusLabel")
        self.summary_label = QLabel("帧数 0 | 检测 0 | FPS 0.00 | 输出目录 -")
        self.summary_label.setObjectName("SummaryLabel")
        self.infer_log = QTextEdit()
        self.infer_log.setReadOnly(True)
        self.infer_log.setMaximumHeight(160)
        self.infer_log.setPlaceholderText("推理日志会显示在这里")

        self.eval_list = QListWidget()
        self.eval_filter_edit = QLineEdit("")
        self.eval_filter_edit.setPlaceholderText("筛选评估结果")
        self.eval_detail = QTextEdit()
        self.eval_detail.setReadOnly(True)
        self.validation_list = QListWidget()
        self.validation_filter_edit = QLineEdit("")
        self.validation_filter_edit.setPlaceholderText("筛选质检报告")
        self.validation_detail = QTextEdit()
        self.validation_detail.setReadOnly(True)
        self.training_list = QListWidget()
        self.training_filter_edit = QLineEdit("")
        self.training_filter_edit.setPlaceholderText("筛选训练结果")
        self.training_detail = QTextEdit()
        self.training_detail.setReadOnly(True)
        self.training_plot_label = QLabel("选择训练结果后显示曲线图")
        self.training_plot_label.setObjectName("PreviewPane")
        self.training_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.training_plot_label.setMinimumHeight(300)
        self.prev_plot_button = QPushButton("上一张图")
        self.next_plot_button = QPushButton("下一张图")

        self.desktop_task_combo = QComboBox()
        self.desktop_task_combo.addItems(
            [
                "导入数据集",
                "数据转换",
                "数据质检",
                "模型评估",
                "模型训练",
                "项目重置",
                "训练曲线生成",
                "列出模型",
                "数据标注",
                "自动标注",
                "AI 任务",
            ]
        )
        self.task_stack = QStackedWidget()

        self.model_catalog_list = QListWidget()
        self.model_catalog_filter_edit = QLineEdit("")
        self.model_catalog_filter_edit.setPlaceholderText("按名称/描述筛选模型")
        # 系列下拉项由 _sync_model_family_combos() 从 model_catalog 动态填充
        self.model_catalog_family_combo = QComboBox()
        self.model_catalog_recommend_edit = QLineEdit("")
        self.model_catalog_recommend_edit.setPlaceholderText("如：最快 / 最准 / yolov8 最准")
        self.model_catalog_recommend_button = QPushButton("推荐")
        self.model_catalog_apply_train_button = QPushButton("应用到模型训练")
        self.model_catalog_apply_infer_button = QPushButton("应用到模型推理")
        self.model_catalog_detail = QTextEdit()
        self.model_catalog_detail.setReadOnly(True)

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

        self.eval_model_edit = QLineEdit(str(_default_model()))
        self.eval_dataset_edit = QLineEdit("steel-surface-defect")
        self.eval_config_edit = QLineEdit("val")
        self.eval_device_edit = QLineEdit("0")
        self.eval_executor_edit = QLineEdit("")
        self.eval_name_edit = QLineEdit("desktop-eval")
        self.eval_extra_args_edit = QLineEdit("")
        self.eval_extra_args_edit.setPlaceholderText("例如 --split val --plots")

        self.train_model_edit = QLineEdit(str(_default_model()))
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

        self.plot_csv_edit = QLineEdit(str(_default_results_csv()))
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

        self.agent_input = QTextEdit()
        self.agent_input.setPlaceholderText("用自然语言描述目标检测任务，如：用最快的模型训练 rsod 数据集")
        self.agent_input.setMaximumHeight(90)
        self.agent_output = QTextEdit()
        self.agent_output.setReadOnly(True)
        self.agent_send_button = QPushButton("发送")
        self.agent_send_button.setProperty("primary", True)
        self.agent_stop_button = QPushButton("停止")
        self.agent_stop_button.setEnabled(False)
        self.agent_base_url_edit = QLineEdit("")
        self.agent_base_url_edit.setPlaceholderText("OpenAI 兼容 API 地址（必填），如 https://api.deepseek.com/v1")
        self.agent_model_edit = QLineEdit("")
        self.agent_model_edit.setPlaceholderText("模型名（必填），如 deepseek-chat")
        self.agent_api_key_edit = QLineEdit("")
        self.agent_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.agent_api_key_edit.setPlaceholderText("API 密钥；留空读环境变量 OPENAI_API_KEY")
        self.agent_max_iterations_spin = QSpinBox()
        self.agent_max_iterations_spin.setRange(1, 100)
        self.agent_max_iterations_spin.setValue(8)
        self.agent_dry_run_check = QCheckBox("安全模式：训练/推理以计划模式执行")
        self.agent_dry_run_check.setChecked(True)

        self.task_command_preview = QTextEdit()
        self.task_command_preview.setReadOnly(True)
        self.task_command_preview.setMaximumHeight(100)
        self.task_output = QTextEdit()
        self.task_output.setReadOnly(True)
        self.task_start_button = QPushButton("启动任务")
        self.task_start_button.setProperty("primary", True)
        self.task_stop_button = QPushButton("停止任务")
        self.task_stop_button.setEnabled(False)

        self._build_layout()
        self._connect_signals()
        self._refresh_all_result_tabs()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name.
        self._stop_worker()
        self._stop_task()
        super().closeEvent(event)

    def _build_layout(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.addWidget(self._build_header())

        tabs = QTabWidget()
        self.tabs = tabs
        tabs.addTab(self._build_inference_tab(), "推理")
        tabs.addTab(
            self._build_browser_tab(
                title="模型评估",
                hint="浏览 odp-val 生成的模型评估审计。",
                list_widget=self.eval_list,
                filter_edit=self.eval_filter_edit,
                detail_widget=self.eval_detail,
                refresh_callback=self._refresh_evaluation_results,
                open_callback=self._open_selected_eval,
            ),
            "模型评估",
        )
        tabs.addTab(
            self._build_browser_tab(
                title="数据质检",
                hint="浏览 odp-validate 生成的报告和整改清单。",
                list_widget=self.validation_list,
                filter_edit=self.validation_filter_edit,
                detail_widget=self.validation_detail,
                refresh_callback=self._refresh_validation_reports,
                open_callback=self._open_selected_validation,
            ),
            "数据质检",
        )
        tabs.addTab(self._build_agent_tab(), "AI 助手")
        tabs.addTab(self._build_model_catalog_tab(), "模型目录")
        tabs.addTab(self._build_training_tab(), "训练结果")
        tabs.addTab(self._build_tasks_tab(), "任务启动")
        root.addWidget(tabs)
        self.setCentralWidget(central)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("Header")
        layout = QHBoxLayout(header)
        title = QLabel("ODPlatform Studio")
        title.setObjectName("AppTitle")
        subtitle = QLabel("目标检测开发平台 | 推理、评估、质检、训练、AI 助手、任务启动")
        subtitle.setObjectName("AppSubtitle")
        text = QVBoxLayout()
        text.addWidget(title)
        text.addWidget(subtitle)
        layout.addLayout(text)
        layout.addStretch(1)
        layout.addWidget(self.status_label)
        return header

    def _build_inference_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_preview_panel())
        splitter.addWidget(self._build_control_panel())
        splitter.setSizes([960, 420])
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.addWidget(splitter)
        return page

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self.image_label, stretch=1)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.infer_log)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        root = QVBoxLayout(panel)
        form_group = QGroupBox("推理参数")
        form = QFormLayout(form_group)
        form.addRow(
            "模型",
            _with_buttons(
                self.model_edit,
                [("选择权重", self._browse_model), ("模型目录", self._open_model_catalog)],
            ),
        )
        form.addRow(
            "输入源",
            _with_buttons(
                self.source_edit,
                [
                    ("图片/视频", self._browse_media_source),
                    ("文件夹", self._browse_folder_source),
                    ("摄像头 0", self._use_camera),
                ],
            ),
        )
        form.addRow("运行配置", _with_buttons(self.runtime_edit, [("选择 YAML", self._browse_runtime)]))
        form.addRow("Pipeline", _with_buttons(self.pipeline_edit, [("选择 YAML", self._browse_pipeline)]))
        form.addRow("任务", self.task_combo)
        form.addRow("置信度", self.conf_edit)
        form.addRow("IoU", self.iou_edit)
        form.addRow("图像尺寸", self.imgsz_spin)
        form.addRow("最大检测数", self.max_det_spin)
        form.addRow("类别过滤", self.classes_edit)
        form.addRow("设备", self.device_edit)
        form.addRow("运行名", self.name_edit)
        form.addRow("最多帧数", self.max_frames_spin)
        form.addRow("视频抽帧间隔", self.vid_stride_spin)
        form.addRow("", self.save_check)
        form.addRow("", self.threaded_check)

        controls = QGridLayout()
        controls.addWidget(self.start_button, 0, 0)
        controls.addWidget(self.pause_button, 0, 1)
        controls.addWidget(self.stop_button, 1, 0)
        controls.addWidget(self.open_output_button, 1, 1)

        root.addWidget(form_group)
        root.addLayout(controls)
        root.addStretch(1)
        return panel

    def _build_browser_tab(
        self,
        *,
        title: str,
        hint: str,
        list_widget: QListWidget,
        filter_edit: QLineEdit,
        detail_widget: QTextEdit,
        refresh_callback,
        open_callback,
    ) -> QWidget:
        refresh_button = QPushButton("刷新")
        open_button = QPushButton("打开所在目录")
        refresh_button.clicked.connect(refresh_callback)
        open_button.clicked.connect(open_callback)

        header = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("SectionTitle")
        hint_label = QLabel(hint)
        hint_label.setObjectName("HintText")
        header.addWidget(title_label)
        header.addWidget(hint_label)

        buttons = QHBoxLayout()
        buttons.addWidget(refresh_button)
        buttons.addWidget(open_button)
        buttons.addStretch(1)

        left = QVBoxLayout()
        left.addLayout(header)
        left.addWidget(filter_edit)
        left.addLayout(buttons)
        left.addWidget(list_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        list_container = QWidget()
        list_container.setLayout(left)
        splitter.addWidget(list_container)
        splitter.addWidget(detail_widget)
        splitter.setSizes([420, 820])

        page = QWidget()
        root = QHBoxLayout(page)
        root.addWidget(splitter)
        return page

    def _build_agent_tab(self) -> QWidget:
        title_label = QLabel("AI 助手")
        title_label.setObjectName("SectionTitle")
        hint_label = QLabel("用自然语言驱动平台执行目标检测任务（Agent 工具编排）。")
        hint_label.setObjectName("HintText")

        config_group = QGroupBox("API 配置")
        config_form = QFormLayout(config_group)
        config_form.addRow("API 地址", self.agent_base_url_edit)
        config_form.addRow("模型名", self.agent_model_edit)
        config_form.addRow("API 密钥", self.agent_api_key_edit)
        config_row = QWidget()
        config_row_layout = QHBoxLayout(config_row)
        config_row_layout.setContentsMargins(0, 0, 0, 0)
        config_row_layout.addWidget(self.agent_max_iterations_spin)
        config_row_layout.addWidget(self.agent_dry_run_check)
        config_form.addRow("迭代上限", config_row)

        controls = QHBoxLayout()
        controls.addWidget(self.agent_send_button)
        controls.addWidget(self.agent_stop_button)
        controls.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(title_label)
        left.addWidget(hint_label)
        left.addWidget(config_group)
        left.addWidget(QLabel("输入任务"))
        left.addWidget(self.agent_input)
        left.addLayout(controls)

        right = QVBoxLayout()
        output_title = QLabel("对话输出")
        output_title.setObjectName("SectionTitle")
        right.addWidget(output_title)
        right.addWidget(self.agent_output)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_container = QWidget()
        left_container.setLayout(left)
        right_container = QWidget()
        right_container.setLayout(right)
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([480, 780])

        page = QWidget()
        root = QHBoxLayout(page)
        root.addWidget(splitter)
        return page

    def _build_model_catalog_tab(self) -> QWidget:
        refresh_button = QPushButton("刷新")
        recommend_row = QWidget()
        recommend_layout = QHBoxLayout(recommend_row)
        recommend_layout.setContentsMargins(0, 0, 0, 0)
        recommend_layout.addWidget(QLabel("推荐"))
        recommend_layout.addWidget(self.model_catalog_recommend_edit)
        recommend_layout.addWidget(self.model_catalog_recommend_button)

        apply_row = QWidget()
        apply_layout = QHBoxLayout(apply_row)
        apply_layout.setContentsMargins(0, 0, 0, 0)
        apply_layout.addWidget(self.model_catalog_apply_train_button)
        apply_layout.addWidget(self.model_catalog_apply_infer_button)
        apply_layout.addStretch(1)

        title_label = QLabel("模型目录")
        title_label.setObjectName("SectionTitle")
        hint_label = QLabel("浏览内置 YOLO 系列模型，支持自然语言推荐，可应用到训练/推理。")
        hint_label.setObjectName("HintText")

        left = QVBoxLayout()
        left.addWidget(title_label)
        left.addWidget(hint_label)
        left.addWidget(recommend_row)
        filter_row = QWidget()
        filter_layout = QHBoxLayout(filter_row)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(QLabel("系列"))
        filter_layout.addWidget(self.model_catalog_family_combo)
        filter_layout.addWidget(self.model_catalog_filter_edit)
        filter_layout.addWidget(refresh_button)
        left.addWidget(filter_row)
        left.addWidget(apply_row)
        left.addWidget(self.model_catalog_list)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        list_container = QWidget()
        list_container.setLayout(left)
        splitter.addWidget(list_container)
        splitter.addWidget(self.model_catalog_detail)
        splitter.setSizes([420, 820])

        page = QWidget()
        root = QHBoxLayout(page)
        root.addWidget(splitter)
        return page

    def _refresh_model_catalog(self) -> None:
        from od_platform.model_catalog import list_models, recommend_model

        self._sync_model_family_combos()

        family = self.model_catalog_family_combo.currentText()
        family = None if family == "全部" else family

        if self.model_catalog_recommend_edit.text().strip():
            models = recommend_model(
                self.model_catalog_recommend_edit.text().strip(),
                family=family,
                limit=20,
            )
        else:
            models = list_models(family=family)

        query = self.model_catalog_filter_edit.text().strip().lower()
        if query:
            models = [info for info in models if query in info.name.lower() or query in info.description.lower()]

        self.model_catalog_list.clear()
        for info in models:
            item_text = f"{info.name}  [{info.family} {info.variant} | {info.size_category}]"
            self.model_catalog_list.addItem(item_text)
            self.model_catalog_list.item(self.model_catalog_list.count() - 1).setData(
                Qt.ItemDataRole.UserRole, info.name
            )
        if self.model_catalog_list.count():
            self.model_catalog_list.setCurrentRow(0)
        else:
            self.model_catalog_detail.setPlainText("没有匹配的模型。")
        self._set_status("就绪", f"模型目录 {len(models)} 个")

    def _sync_model_family_combos(self) -> None:
        """从 model_catalog 动态同步系列下拉项，避免与目录硬编码重复。"""
        from od_platform.model_catalog import list_families

        families = list_families()
        for combo in (self.model_catalog_family_combo, self.list_models_family_combo):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(["全部", *families])
            combo.setCurrentText(current if current in ("全部", *families) else "全部")
            combo.blockSignals(False)

    def _show_model_catalog_item(self, item) -> None:
        if item is None:
            return
        from od_platform.model_catalog import get_model_info

        info = get_model_info(item.data(Qt.ItemDataRole.UserRole))
        if info is None:
            self.model_catalog_detail.setPlainText("无法读取模型元数据。")
            return
        metric_lines = "\n".join(f"{key:<12}: {value}" for key, value in info.metrics.items())
        lines = [
            "模型详情",
            "=" * 60,
            f"名称          : {info.name}",
            f"系列          : {info.family}",
            f"变体          : {info.variant}",
            f"任务          : {info.task}",
            f"后端          : {info.backend}",
            f"尺寸档位      : {info.size_category}",
            f"参数量        : {info.params_m} M",
            f"CPU 耗时      : {info.speed_cpu_ms} ms",
            "",
            "指标",
            "-" * 60,
            metric_lines,
            "",
            "描述",
            "-" * 60,
            info.description,
        ]
        self.model_catalog_detail.setPlainText("\n".join(lines))

    def _apply_model_to_train(self) -> None:
        item = self.model_catalog_list.currentItem()
        if item is None:
            self._set_status("提示", "先在模型目录中选择一个模型")
            return
        model_name = item.data(Qt.ItemDataRole.UserRole)
        self.train_model_edit.setText(model_name)
        self.tabs.setCurrentIndex(self._task_tab_index())
        self.desktop_task_combo.setCurrentText("模型训练")
        self._set_status("就绪", f"已应用到模型训练: {model_name}")

    def _apply_model_to_infer(self) -> None:
        item = self.model_catalog_list.currentItem()
        if item is None:
            self._set_status("提示", "先在模型目录中选择一个模型")
            return
        model_name = item.data(Qt.ItemDataRole.UserRole)
        self.model_edit.setText(model_name)
        self.tabs.setCurrentIndex(0)  # 推理 tab
        self._set_status("就绪", f"已应用到模型推理: {model_name}")

    def _build_training_tab(self) -> QWidget:
        refresh_button = QPushButton("刷新")
        open_button = QPushButton("打开训练目录")
        refresh_button.clicked.connect(self._refresh_training_results)
        open_button.clicked.connect(self._open_selected_training)
        self.prev_plot_button.clicked.connect(self._show_previous_training_plot)
        self.next_plot_button.clicked.connect(self._show_next_training_plot)

        title_label = QLabel("训练结果")
        title_label.setObjectName("SectionTitle")
        hint_label = QLabel("浏览 YOLO 训练 run、最后一轮指标、权重和训练曲线。")
        hint_label.setObjectName("HintText")

        buttons = QHBoxLayout()
        buttons.addWidget(refresh_button)
        buttons.addWidget(open_button)
        buttons.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(title_label)
        left.addWidget(hint_label)
        left.addWidget(self.training_filter_edit)
        left.addLayout(buttons)
        left.addWidget(self.training_list)

        plot_buttons = QHBoxLayout()
        plot_buttons.addWidget(self.prev_plot_button)
        plot_buttons.addWidget(self.next_plot_button)
        plot_buttons.addStretch(1)

        right = QVBoxLayout()
        right.addWidget(self.training_detail, stretch=1)
        right.addWidget(self.training_plot_label, stretch=1)
        right.addLayout(plot_buttons)

        list_container = QWidget()
        list_container.setLayout(left)
        detail_container = QWidget()
        detail_container.setLayout(right)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(list_container)
        splitter.addWidget(detail_container)
        splitter.setSizes([360, 900])

        page = QWidget()
        root = QHBoxLayout(page)
        root.addWidget(splitter)
        return page

    def _build_tasks_tab(self) -> QWidget:
        task_group = QGroupBox("选择任务")
        task_form = QFormLayout(task_group)
        task_form.addRow("任务类型", self.desktop_task_combo)

        self.task_stack.addWidget(
            _form_page(
                "导入数据集参数",
                [
                    ("数据集名称", self.import_dataset_edit),
                    ("数据 zip", _with_buttons(self.import_zip_edit, [("选择 zip", self._browse_import_zip)])),
                    ("标注格式", self.import_format_combo),
                    ("", self.import_overwrite_check),
                    ("追加参数", self.import_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            _form_page(
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
            _form_page(
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
            _form_page(
                "模型评估参数",
                [
                    ("模型权重", _with_buttons(self.eval_model_edit, [("选择权重", self._browse_eval_model)])),
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
            _form_page(
                "模型训练参数",
                [
                    ("模型权重/名称", _with_buttons(self.train_model_edit, [("选择权重", self._browse_train_model)])),
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
            _form_page(
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
            _form_page(
                "训练曲线生成参数",
                [
                    ("results.csv", _with_buttons(self.plot_csv_edit, [("选择 CSV", self._browse_plot_csv)])),
                    ("输出图片", _with_buttons(self.plot_output_edit, [("选择 PNG", self._browse_plot_output)])),
                    ("摘要 JSON", _with_buttons(self.plot_summary_edit, [("选择 JSON", self._browse_plot_summary)])),
                    ("", self.plot_matplotx_check),
                    ("追加参数", self.plot_extra_args_edit),
                ],
            )
        )
        self.task_stack.addWidget(
            _form_page(
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
            _form_page(
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
            _form_page(
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
            _form_page(
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

        page = QWidget()
        root = QHBoxLayout(page)
        root.addWidget(splitter)
        return page

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._start_worker)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop_worker)
        self.open_output_button.clicked.connect(self._open_last_output_dir)
        self.eval_list.currentItemChanged.connect(lambda current, _previous: self._show_eval_item(current))
        self.validation_list.currentItemChanged.connect(lambda current, _previous: self._show_validation_item(current))
        self.training_list.currentItemChanged.connect(lambda current, _previous: self._show_training_item(current))
        self.model_catalog_list.currentItemChanged.connect(lambda current, _previous: self._show_model_catalog_item(current))
        self.model_catalog_recommend_button.clicked.connect(self._refresh_model_catalog)
        self.model_catalog_apply_train_button.clicked.connect(self._apply_model_to_train)
        self.model_catalog_apply_infer_button.clicked.connect(self._apply_model_to_infer)
        self.model_catalog_filter_edit.textChanged.connect(lambda _text: self._refresh_model_catalog())
        self.model_catalog_family_combo.currentTextChanged.connect(lambda _text: self._refresh_model_catalog())
        self.eval_filter_edit.textChanged.connect(lambda _text: self._refresh_evaluation_results())
        self.validation_filter_edit.textChanged.connect(lambda _text: self._refresh_validation_reports())
        self.training_filter_edit.textChanged.connect(lambda _text: self._refresh_training_results())
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
        self.task_start_button.clicked.connect(self._start_task)
        self.task_stop_button.clicked.connect(self._stop_task)
        self.agent_send_button.clicked.connect(self._start_agent)
        self.agent_stop_button.clicked.connect(self._stop_agent)
        # AI 助手页与 AI 任务页共享 API 配置（双向同步，防重复填写）
        self._sync_agent_config_fields()
        self._switch_task_page(self.desktop_task_combo.currentIndex())
        self._refresh_task_preview()
        self._refresh_model_catalog()

    @Slot()
    def _start_worker(self) -> None:
        if self._thread is not None:
            return
        try:
            conf = float(self.conf_edit.text().strip())
            iou = float(self.iou_edit.text().strip())
        except ValueError:
            self._set_status("失败", "置信度和 IoU 必须是数字")
            return

        self.infer_log.clear()
        self._append_log("启动推理任务")
        self._append_log(f"模型: {self.model_edit.text().strip()}")
        self._append_log(f"输入源: {self.source_edit.text().strip()}")

        self._thread = QThread(self)
        self._worker = InferWorker(
            model=self.model_edit.text().strip(),
            source=self.source_edit.text().strip(),
            conf=conf,
            iou=iou,
            device=self.device_edit.text().strip() or None,
            name=self.name_edit.text().strip() or "desktop-infer",
            task=self.task_combo.currentText(),
            imgsz=self.imgsz_spin.value(),
            max_det=self.max_det_spin.value(),
            classes=self.classes_edit.text().strip() or None,
            runtime_config=self.runtime_edit.text().strip() or None,
            pipeline_yaml=self.pipeline_edit.text().strip() or None,
            save_outputs=self.save_check.isChecked(),
            threaded=self.threaded_check.isChecked(),
            max_frames=self.max_frames_spin.value() or None,
            vid_stride=self.vid_stride_spin.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._show_frame)
        self._worker.progress_ready.connect(self._show_progress)
        self._worker.completed.connect(self._finish_success)
        self._worker.failed.connect(self._finish_failed)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.pause_button.setText("暂停")
        self.stop_button.setEnabled(True)
        self._set_status("运行中", "推理运行中")
        self._thread.start()

    @Slot()
    def _toggle_pause(self) -> None:
        if self._worker is None:
            return
        paused = self._worker.toggle_pause()
        if paused:
            self.pause_button.setText("继续")
            self._set_status("已暂停", "推理已暂停")
            self._append_log("推理已暂停")
        else:
            self.pause_button.setText("暂停")
            self._set_status("运行中", "推理运行中")
            self._append_log("推理已继续")

    @Slot()
    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._set_status("停止中", "正在停止")
            self._append_log("请求停止推理")

    @Slot(object, object)
    def _show_frame(self, _frame, annotated) -> None:
        pixmap = _to_pixmap(annotated)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    @Slot(object)
    def _show_progress(self, event) -> None:
        total = event.total_frames if event.total_frames is not None else "未知"
        self._set_status("运行中", f"帧 {event.frame_index}/{total}，FPS {event.loop_fps:.2f}")
        self.summary_label.setText(
            f"帧数 {event.frame_index} | 检测 {event.detections_total} | FPS {event.loop_fps:.2f} | 输出目录 {self._last_output_dir or '-'}"
        )
        if event.frame_index % 30 == 0:
            self._append_log(f"进度: frame={event.frame_index}, fps={event.loop_fps:.2f}, detections={event.detections_total}")

    @Slot(object)
    def _finish_success(self, result) -> None:
        self._last_output_dir = Path(result.output_dir)
        self.open_output_button.setEnabled(True)
        stats = result.stats or {}
        self.summary_label.setText(
            " | ".join(
                [
                    f"帧数 {stats.get('frames', 0)}",
                    f"检测 {stats.get('detections', 0)}",
                    f"平均 FPS {stats.get('avg_fps', 0)}",
                    f"输出目录 {result.output_dir}",
                ]
            )
        )
        self._set_status("完成", f"输出目录 {result.output_dir}")
        self._append_log(f"推理完成: {json.dumps(stats, ensure_ascii=False)}")

    @Slot(str)
    def _finish_failed(self, message: str) -> None:
        self._set_status("失败", message)
        self._append_log(f"推理失败: {message}")

    @Slot()
    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self.stop_button.setEnabled(False)
        self._refresh_all_result_tabs()

    def _switch_task_page(self, index: int) -> None:
        self.task_stack.setCurrentIndex(max(0, index))

    @Slot()
    def _start_task(self) -> None:
        if self._task_thread is not None:
            return
        try:
            module, args = self._build_task_command()
        except ValueError as exc:
            self.task_output.append(f"参数错误: {exc}")
            self._set_status("失败", str(exc))
            return

        self.task_output.clear()
        self.task_output.append("启动桌面端任务")
        self._task_thread = QThread(self)
        self._task_worker = CommandWorker(module=module, args=args, cwd=ROOT_DIR, platform_src=PLATFORM_SRC)
        self._task_worker.moveToThread(self._task_thread)
        self._task_thread.started.connect(self._task_worker.run)
        self._task_worker.output_ready.connect(self.task_output.append)
        self._task_worker.completed.connect(self._finish_task)
        self._task_worker.failed.connect(self._fail_task)
        self._task_worker.completed.connect(self._task_thread.quit)
        self._task_worker.failed.connect(self._task_thread.quit)
        self._task_thread.finished.connect(self._cleanup_task_thread)
        self.task_start_button.setEnabled(False)
        self.task_stop_button.setEnabled(True)
        self._set_status("运行中", f"任务启动: {self.desktop_task_combo.currentText()}")
        self._task_thread.start()

    @Slot()
    def _stop_task(self) -> None:
        if self._task_worker is not None:
            self._task_worker.cancel()
            self._set_status("停止中", "正在停止桌面端任务")

    @Slot(int)
    def _finish_task(self, exit_code: int) -> None:
        self.task_output.append(f"任务结束，退出码: {exit_code}")
        if exit_code == 0:
            self._set_status("完成", "桌面端任务完成")
        else:
            self._set_status("失败", f"桌面端任务退出码 {exit_code}")

    @Slot(str)
    def _fail_task(self, message: str) -> None:
        self.task_output.append(f"任务失败: {message}")
        self._set_status("失败", message)

    @Slot()
    def _cleanup_task_thread(self) -> None:
        if self._task_worker is not None:
            self._task_worker.deleteLater()
        if self._task_thread is not None:
            self._task_thread.deleteLater()
        self._task_worker = None
        self._task_thread = None
        self.task_start_button.setEnabled(True)
        self.task_stop_button.setEnabled(False)
        self._refresh_all_result_tabs()

    def _sync_agent_config_fields(self) -> None:
        """AI 助手页与 AI 任务页的 API 配置字段双向同步（防重复维护）。"""
        self._syncing_agent_config = False

        def _link(source, target) -> None:
            def _handler(text: str) -> None:
                if self._syncing_agent_config:
                    return
                self._syncing_agent_config = True
                target.setText(text)
                self._syncing_agent_config = False

            source.textChanged.connect(_handler)

        _link(self.agent_base_url_edit, self.ai_task_base_url_edit)
        _link(self.ai_task_base_url_edit, self.agent_base_url_edit)
        _link(self.agent_model_edit, self.ai_task_model_edit)
        _link(self.ai_task_model_edit, self.agent_model_edit)
        _link(self.agent_api_key_edit, self.ai_task_api_key_edit)
        _link(self.ai_task_api_key_edit, self.agent_api_key_edit)

    @Slot()
    def _start_agent(self) -> None:
        """启动一轮 Agent 对话（QThread 后台执行）。"""
        if self._agent_thread is not None:
            return
        message = self.agent_input.toPlainText().strip()
        base_url = self.agent_base_url_edit.text().strip()
        model = self.agent_model_edit.text().strip()
        api_key = self.agent_api_key_edit.text().strip() or None
        if not message:
            self._set_status("提示", "请输入任务描述")
            return
        if not base_url or not model:
            self._set_status("提示", "请填写 API 地址与模型名")
            return

        self.agent_output.append(f"> {message}")
        self.agent_send_button.setEnabled(False)
        self.agent_stop_button.setEnabled(True)
        self._set_status("运行中", f"AI 助手: {model}")

        self._agent_thread = QThread(self)
        self._agent_worker = AgentWorker(
            model=model,
            api_key=api_key,
            base_url=base_url,
            message=message,
            max_iterations=self.agent_max_iterations_spin.value(),
            dry_run=self.agent_dry_run_check.isChecked(),
        )
        self._agent_worker.moveToThread(self._agent_thread)
        self._agent_thread.started.connect(self._agent_worker.run)
        self._agent_worker.event_ready.connect(self._on_agent_event)
        self._agent_worker.completed.connect(self._finish_agent)
        self._agent_worker.cancelled.connect(self._cancel_agent)
        self._agent_worker.failed.connect(self._fail_agent)
        self._agent_worker.completed.connect(self._agent_thread.quit)
        self._agent_worker.cancelled.connect(self._agent_thread.quit)
        self._agent_worker.failed.connect(self._agent_thread.quit)
        self._agent_thread.finished.connect(self._cleanup_agent_thread)
        self._agent_thread.start()

    @Slot()
    def _stop_agent(self) -> None:
        if self._agent_worker is not None:
            self._agent_worker.cancel()
            self._set_status("停止中", "正在停止 AI 助手")

    @Slot(object)
    def _on_agent_event(self, event: object) -> None:
        """渲染 AgentEvent（message/tool_start/tool_result/error/done）。"""
        kind = getattr(event, "kind", "")
        content = getattr(event, "content", None)
        if kind == "message" and content:
            self.agent_output.append(f"🤖 {content}")
        elif kind == "tool_start":
            self.agent_output.append(f"→ 调用工具 {event.tool_name}({content or ''})")
        elif kind == "tool_result" and event.tool_result is not None:
            status = "成功" if event.tool_result.ok else "失败"
            summary = (event.tool_result.summary or "")[:200]
            self.agent_output.append(f"← {event.tool_name} {status}: {summary}")
        elif kind == "error":
            self.agent_output.append(f"❌ 错误: {content}")
        elif kind == "done":
            self.agent_output.append("── 完成 ──")

    @Slot(str)
    def _finish_agent(self, final_text: str) -> None:
        if final_text:
            self.agent_output.append(f"🤖 {final_text}")
        self._set_status("完成", "AI 助手对话完成")

    @Slot()
    def _cancel_agent(self) -> None:
        self.agent_output.append("⏹ 已取消")
        self._set_status("已取消", "AI 助手对话已取消")

    @Slot(str)
    def _fail_agent(self, message: str) -> None:
        self.agent_output.append(f"❌ 失败: {message}")
        self._set_status("失败", message)

    @Slot()
    def _cleanup_agent_thread(self) -> None:
        if self._agent_worker is not None:
            self._agent_worker.deleteLater()
        if self._agent_thread is not None:
            self._agent_thread.deleteLater()
        self._agent_worker = None
        self._agent_thread = None
        self.agent_send_button.setEnabled(True)
        self.agent_stop_button.setEnabled(False)

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

    def _build_task_command(self) -> tuple[str, list[str]]:
        task_name = self.desktop_task_combo.currentText()

        if task_name == "导入数据集":
            dataset = self.import_dataset_edit.text().strip()
            zip_path = self.import_zip_edit.text().strip()
            extra_args = _split_extra_args(self.import_extra_args_edit.text().strip())
            if not zip_path:
                raise ValueError("导入数据集需要填写数据 zip")
            args = [zip_path]
            if dataset:
                args.extend(["--name", dataset])
            args.extend(["--format", self.import_format_combo.currentText()])
            if self.import_overwrite_check.isChecked():
                args.append("--overwrite")
            return "od_platform.cli.import_dataset", [*args, *extra_args]

        if task_name == "数据转换":
            dataset = self.transform_dataset_edit.text().strip()
            extra_args = _split_extra_args(self.transform_extra_args_edit.text().strip())
            if not dataset:
                raise ValueError("数据转换需要填写数据集名称")
            annotation_format = self.transform_format_combo.currentText()
            if annotation_format == "voc":
                annotation_format = "pascal_voc"
            args = [
                "--dataset",
                dataset,
                "--format",
                annotation_format,
                "--task",
                self.transform_task_combo.currentText(),
            ]
            return "od_platform.cli.transform_data", [*args, *extra_args]

        if task_name == "数据质检":
            dataset = self.validate_dataset_edit.text().strip()
            executor = self.validate_executor_edit.text().strip()
            extra_args = _split_extra_args(self.validate_extra_args_edit.text().strip())
            if not dataset:
                raise ValueError("数据质检需要填写数据集名称")
            args = ["--dataset", dataset, "--task", self.validate_task_combo.currentText()]
            if executor:
                args.extend(["--executor", executor])
            return "od_platform.cli.validate_data", [*args, *extra_args]

        if task_name == "模型评估":
            model = self.eval_model_edit.text().strip()
            dataset = self.eval_dataset_edit.text().strip()
            device = self.eval_device_edit.text().strip()
            executor = self.eval_executor_edit.text().strip()
            run_name = self.eval_name_edit.text().strip()
            extra_args = _split_extra_args(self.eval_extra_args_edit.text().strip())
            if not model:
                raise ValueError("模型评估需要填写模型权重")
            if not dataset:
                raise ValueError("模型评估需要填写数据集名称")
            args = ["--config", self.eval_config_edit.text().strip() or "val", "--model", model, "--data", dataset]
            if device:
                args.extend(["--device", device])
            if executor:
                args.extend(["--executor", executor])
            if run_name:
                args.extend(["--name", run_name])
            return "od_platform.cli.evaluate_model", [*args, *extra_args]

        if task_name == "模型训练":
            model = self.train_model_edit.text().strip()
            dataset = self.train_dataset_edit.text().strip()
            device = self.train_device_edit.text().strip()
            executor = self.train_executor_edit.text().strip()
            run_name = self.train_name_edit.text().strip()
            extra_args = _split_extra_args(self.train_extra_args_edit.text().strip())
            if not model:
                raise ValueError("模型训练需要填写模型权重或模型名")
            if not dataset:
                raise ValueError("模型训练需要填写数据集名称")
            args = [
                "--config",
                self.train_config_edit.text().strip() or "train",
                "--model",
                model,
                "--data",
                dataset,
                "--epochs",
                str(self.train_epochs_spin.value()),
                "--batch",
                str(self.train_batch_spin.value()),
                "--workers",
                str(self.train_workers_spin.value()),
            ]
            if device:
                args.extend(["--device", device])
            if executor:
                args.extend(["--executor", executor])
            if run_name:
                args.extend(["--name", run_name])
            if self.train_dry_run_check.isChecked():
                args.append("--dry-run")
            return "od_platform.cli.train_model", [*args, *extra_args]

        if task_name == "项目重置":
            args: list[str] = []
            extra_args = _split_extra_args(self.reset_extra_args_edit.text().strip())
            if self.reset_dry_run_check.isChecked():
                args.append("--dry-run")
            if self.reset_backup_check.isChecked():
                args.append("--backup")
            if self.reset_yes_check.isChecked():
                args.append("--yes")
            if self.reset_force_check.isChecked():
                args.append("--force")
            return "od_platform.cli.reset_project", [*args, *extra_args]

        if task_name == "训练曲线生成":
            csv_path = self.plot_csv_edit.text().strip()
            output_path = self.plot_output_edit.text().strip()
            summary_path = self.plot_summary_edit.text().strip()
            extra_args = _split_extra_args(self.plot_extra_args_edit.text().strip())
            if not csv_path:
                raise ValueError("训练曲线生成需要填写 results.csv")
            args = [csv_path]
            if output_path:
                args.extend(["--output", output_path])
            if summary_path:
                args.extend(["--summary", summary_path])
            if self.plot_matplotx_check.isChecked():
                args.append("--matplotx")
            return "od_platform.cli.plot_training", [*args, *extra_args]

        if task_name == "列出模型":
            extra_args = _split_extra_args(self.list_models_extra_args_edit.text().strip())
            args: list[str] = []
            family = self.list_models_family_combo.currentText()
            if family and family != "全部":
                args.extend(["--family", family])
            recommend = self.list_models_recommend_edit.text().strip()
            if recommend:
                args.extend(["--recommend", recommend])
            args.extend(["--limit", str(self.list_models_limit_spin.value())])
            if self.list_models_json_check.isChecked():
                args.append("--json")
            return "od_platform.model_catalog.cli.list_models", [*args, *extra_args]

        if task_name == "数据标注":
            dataset = self.annotate_dataset_edit.text().strip()
            classes = self.annotate_classes_edit.text().split()
            extra_args = _split_extra_args(self.annotate_extra_args_edit.text().strip())
            if not dataset:
                raise ValueError("数据标注需要填写数据集名称")
            if not classes:
                raise ValueError("数据标注需要填写至少一个类别名")
            args = ["--dataset", dataset, "--classes", *classes]
            if self.annotate_resume_check.isChecked():
                args.append("--resume")
            return "od_platform.annotation.cli.annotate", [*args, *extra_args]

        if task_name == "自动标注":
            dataset = self.auto_annotate_dataset_edit.text().strip()
            classes = self.auto_annotate_classes_edit.text().split()
            prompt = self.auto_annotate_prompt_edit.text().strip()
            base_url = self.auto_annotate_base_url_edit.text().strip()
            model = self.auto_annotate_model_edit.text().strip()
            api_key = self.auto_annotate_api_key_edit.text().strip()
            extra_args = _split_extra_args(self.auto_annotate_extra_args_edit.text().strip())
            if not dataset:
                raise ValueError("自动标注需要填写数据集名称")
            if not classes:
                raise ValueError("自动标注需要填写至少一个类别名")
            if not base_url:
                raise ValueError("自动标注需要填写 API 地址")
            if not model:
                raise ValueError("自动标注需要填写视觉模型名")
            args = ["--dataset", dataset, "--classes", *classes, "--prompt", prompt, "--base-url", base_url, "--model", model]
            if api_key:
                args.extend(["--api-key", api_key])
            if self.auto_annotate_limit_spin.value():
                args.extend(["--limit", str(self.auto_annotate_limit_spin.value())])
            if self.auto_annotate_dry_run_check.isChecked():
                args.append("--dry-run")
            return "od_platform.annotation.cli.auto_annotate", [*args, *extra_args]

        if task_name == "AI 任务":
            prompt = self.ai_task_prompt_edit.text().strip()
            base_url = self.ai_task_base_url_edit.text().strip()
            model = self.ai_task_model_edit.text().strip()
            api_key = self.ai_task_api_key_edit.text().strip()
            extra_args = _split_extra_args(self.ai_task_extra_args_edit.text().strip())
            if not prompt:
                raise ValueError("AI 任务需要填写自然语言任务描述")
            if not base_url:
                raise ValueError("AI 任务需要填写 API 地址")
            if not model:
                raise ValueError("AI 任务需要填写模型名")
            args = ["--base-url", base_url, "--model", model, "--task", prompt]
            if api_key:
                args.extend(["--api-key", api_key])
            return "od_platform.agent.cli.agent_chat", [*args, *extra_args]

        raise ValueError(f"未知任务: {task_name}")

    def _open_model_catalog(self) -> None:
        """跳转到模型目录标签页并刷新。"""
        self.tabs.setCurrentIndex(self._model_catalog_tab_index())
        self._refresh_model_catalog()

    def _model_catalog_tab_index(self) -> int:
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == "模型目录":
                return index
        return 3  # 兜底：推理/评估/质检之后

    def _task_tab_index(self) -> int:
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == "任务启动":
                return index
        return self.tabs.count() - 1  # 兜底：最后一个 tab

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", str(ROOT_DIR), "PyTorch weights (*.pt);;All files (*)")
        if path:
            self.model_edit.setText(path)

    def _browse_eval_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", str(ROOT_DIR), "PyTorch weights (*.pt);;All files (*)")
        if path:
            self.eval_model_edit.setText(path)

    def _browse_train_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", str(ROOT_DIR), "PyTorch weights (*.pt);;All files (*)")
        if path:
            self.train_model_edit.setText(path)

    def _browse_import_zip(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择数据集 zip", str(ROOT_DIR.parent), "Zip files (*.zip);;All files (*)")
        if path:
            self.import_zip_edit.setText(path)

    def _browse_plot_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 results.csv", str(ROOT_DIR / "runs"), "CSV files (*.csv);;All files (*)")
        if path:
            self.plot_csv_edit.setText(path)

    def _browse_plot_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "选择输出图片", str(ROOT_DIR / "runs" / "training_summary.png"), "PNG files (*.png);;All files (*)")
        if path:
            self.plot_output_edit.setText(path)

    def _browse_plot_summary(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "选择摘要 JSON", str(ROOT_DIR / "runs" / "training_summary.json"), "JSON files (*.json);;All files (*)")
        if path:
            self.plot_summary_edit.setText(path)

    def _browse_media_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片或视频",
            str(ROOT_DIR),
            "Media files (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mkv *.mov *.flv *.wmv);;All files (*)",
        )
        if path:
            self.source_edit.setText(path)

    def _browse_folder_source(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择图片文件夹", str(ROOT_DIR))
        if directory:
            self.source_edit.setText(directory)

    def _browse_runtime(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择推理运行配置", str(ROOT_DIR), "YAML (*.yaml *.yml)")
        if path:
            self.runtime_edit.setText(path)

    def _browse_pipeline(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择推理 pipeline 配置", str(ROOT_DIR), "YAML (*.yaml *.yml)")
        if path:
            self.pipeline_edit.setText(path)

    def _use_camera(self) -> None:
        self.source_edit.setText("0")

    def _open_last_output_dir(self) -> None:
        if self._last_output_dir is not None:
            _open_path(self._last_output_dir)

    def _refresh_all_result_tabs(self) -> None:
        self._refresh_evaluation_results()
        self._refresh_validation_reports()
        self._refresh_training_results()

    def _refresh_evaluation_results(self) -> None:
        paths = sorted((ROOT_DIR / "runs" / "evaluation").glob("**/odp_audit.json"), reverse=True)
        paths = _filter_paths(paths, self.eval_filter_edit.text())
        self._fill_list(self.eval_list, paths)
        if self.eval_list.count() == 0:
            self.eval_detail.setPlainText("暂无模型评估结果。运行 odp-val 后会在这里显示。")

    def _refresh_validation_reports(self) -> None:
        paths = sorted((ROOT_DIR / "runs" / "data_validation").glob("**/report.json"), reverse=True)
        if not paths:
            paths = sorted((ROOT_DIR / "runs" / "data_validation").glob("**/report.md"), reverse=True)
        paths = _filter_paths(paths, self.validation_filter_edit.text())
        self._fill_list(self.validation_list, paths)
        if self.validation_list.count() == 0:
            self.validation_detail.setPlainText("暂无数据质检报告。运行 odp-validate 后会在这里显示。")

    def _refresh_training_results(self) -> None:
        result_dirs = sorted({path.parent for path in (ROOT_DIR / "runs").glob("**/results.csv")}, reverse=True)
        result_dirs = _filter_paths(result_dirs, self.training_filter_edit.text())
        self._fill_list(self.training_list, result_dirs)
        if self.training_list.count() == 0:
            self.training_detail.setPlainText("暂无训练结果。运行 odp-train 后会在这里显示。")
            self.training_plot_label.setText("暂无训练曲线")

    @staticmethod
    def _fill_list(list_widget: QListWidget, paths: list[Path]) -> None:
        list_widget.clear()
        for path in paths:
            item_text = str(path.relative_to(ROOT_DIR)) if _is_relative_to(path, ROOT_DIR) else str(path)
            list_widget.addItem(item_text)
            list_widget.item(list_widget.count() - 1).setData(Qt.ItemDataRole.UserRole, str(path))
        if list_widget.count():
            list_widget.setCurrentRow(0)

    def _show_eval_item(self, item) -> None:
        if item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        payload = _read_json(path)
        metrics = payload.get("metrics", {})
        lines = [
            "模型评估摘要",
            "=" * 60,
            f"运行名      : {payload.get('run_name') or payload.get('audit_run_name')}",
            f"模型        : {payload.get('model_ref')}",
            f"数据        : {payload.get('data_yaml') or payload.get('config', {}).get('data')}",
            f"创建时间    : {payload.get('created_at')}",
            f"耗时        : {payload.get('elapsed_seconds')} 秒",
            "",
            "指标",
            "-" * 60,
        ]
        lines.extend(_format_mapping(metrics, preferred=("fitness", "map50", "map50_95", "precision", "recall")))
        lines.extend(["", "原始 JSON", "-" * 60, json.dumps(payload, ensure_ascii=False, indent=2)])
        self.eval_detail.setPlainText("\n".join(lines))

    def _show_validation_item(self, item) -> None:
        if item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        if path.suffix.lower() != ".json":
            self.validation_detail.setPlainText(path.read_text(encoding="utf-8", errors="replace"))
            return
        payload = _read_json(path)
        counts = payload.get("counts", {})
        summary = payload.get("dataset_summary", {})
        results = payload.get("results", [])
        lines = [
            "数据质检摘要",
            "=" * 60,
            f"运行 ID     : {payload.get('run_id')}",
            f"严重级别    : {payload.get('overall_severity')}",
            f"退出码      : {payload.get('exit_code')}",
            f"数据 YAML   : {payload.get('yaml_path')}",
            f"耗时        : {payload.get('duration_seconds')} 秒",
            "",
            "结果计数",
            "-" * 60,
            *_format_mapping(counts),
            "",
            "数据集概览",
            "-" * 60,
            *_format_mapping(summary),
            "",
            "检查项",
            "-" * 60,
        ]
        for result in results:
            lines.append(f"[{result.get('severity')}] {result.get('name')}: {result.get('summary')}")
        fix_items = payload.get("fix_items") or []
        if fix_items:
            lines.extend(["", "整改清单", "-" * 60])
            lines.extend(f"- {item}" for item in fix_items[:50])
        self.validation_detail.setPlainText("\n".join(lines))

    def _show_training_item(self, item) -> None:
        if item is None:
            return
        run_dir = Path(item.data(Qt.ItemDataRole.UserRole))
        results_csv = run_dir / "results.csv"
        lines = ["训练结果摘要", "=" * 60, f"训练目录: {run_dir}", ""]
        if results_csv.exists():
            rows = _read_csv_rows(results_csv)
            lines.append(f"results.csv: {results_csv}")
            lines.append(f"epoch 数   : {len(rows)}")
            if rows:
                lines.extend(["", "最后一轮指标", "-" * 60])
                last = rows[-1]
                lines.extend(
                    _format_mapping(
                        last,
                        preferred=(
                            "epoch",
                            "train/box_loss",
                            "val/box_loss",
                            "metrics/precision(B)",
                            "metrics/recall(B)",
                            "metrics/mAP50(B)",
                            "metrics/mAP50-95(B)",
                        ),
                    )
                )
        else:
            lines.append("未找到 results.csv。")

        weights = sorted((run_dir / "weights").glob("*.pt"))
        if weights:
            lines.extend(["", "权重文件", "-" * 60])
            lines.extend(f"- {path.name}" for path in weights)
        plots = sorted(run_dir.glob("*.png"))[:20]
        if plots:
            lines.extend(["", "可用图表", "-" * 60])
            lines.extend(f"- {path.name}" for path in plots)
        self._training_plots = plots
        self._training_plot_index = 0
        self._show_training_plot()
        self.training_detail.setPlainText("\n".join(lines))

    def _show_previous_training_plot(self) -> None:
        if not self._training_plots:
            return
        self._training_plot_index = (self._training_plot_index - 1) % len(self._training_plots)
        self._show_training_plot()

    def _show_next_training_plot(self) -> None:
        if not self._training_plots:
            return
        self._training_plot_index = (self._training_plot_index + 1) % len(self._training_plots)
        self._show_training_plot()

    def _show_training_plot(self) -> None:
        if not self._training_plots:
            self.training_plot_label.setText("当前训练目录没有可预览的 PNG 曲线")
            self.training_plot_label.setPixmap(QPixmap())
            return
        path = self._training_plots[self._training_plot_index]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.training_plot_label.setText(f"无法加载图表: {path.name}")
            return
        self.training_plot_label.setPixmap(
            pixmap.scaled(
                self.training_plot_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.training_plot_label.setToolTip(str(path))

    def _open_selected_eval(self) -> None:
        _open_selected_parent(self.eval_list)

    def _open_selected_validation(self) -> None:
        _open_selected_parent(self.validation_list)

    def _open_selected_training(self) -> None:
        item = self.training_list.currentItem()
        if item is not None:
            _open_path(Path(item.data(Qt.ItemDataRole.UserRole)))

    def _set_status(self, state: str, detail: str) -> None:
        self.status_label.setText(f"状态：{state} | {detail}")

    def _append_log(self, message: str) -> None:
        self.infer_log.append(message)


def _open_path(path: Path) -> None:
    target = path if path.is_dir() else path.parent
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


def _open_selected_parent(list_widget: QListWidget) -> None:
    item = list_widget.currentItem()
    if item is not None:
        _open_path(Path(item.data(Qt.ItemDataRole.UserRole)))


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - result viewer should keep the UI alive.
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except UnicodeDecodeError:
        with path.open("r", encoding="gbk", newline="") as handle:
            return list(csv.DictReader(handle))


def _format_mapping(data: Any, preferred: tuple[str, ...] = ()) -> list[str]:
    if not isinstance(data, dict):
        return [str(data)]
    lines: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if key in data:
            lines.append(f"{key:<28}: {data[key]}")
            seen.add(key)
    for key, value in data.items():
        if key in seen:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key:<28}: {value}")
    return lines


def _filter_paths(paths: list[Path], query: str) -> list[Path]:
    query = query.strip().lower()
    if not query:
        return paths
    return [path for path in paths if query in str(path).lower()]


def _split_extra_args(text: str) -> list[str]:
    if not text:
        return []
    try:
        return shlex.split(text, posix=False)
    except ValueError as exc:
        raise ValueError(f"追加参数解析失败: {exc}") from exc


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _with_buttons(line_edit: QLineEdit, buttons: list[tuple[str, object]]) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(line_edit)
    for label, callback in buttons:
        button = QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(button)
    return container


def _form_page(title: str, rows: list[tuple[str, QWidget]]) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    group = QGroupBox(title)
    form = QFormLayout(group)
    for label, widget in rows:
        form.addRow(label, widget)
    layout.addWidget(group)
    layout.addStretch(1)
    return page


def _to_pixmap(image) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    q_image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(q_image)


def _default_model() -> Path:
    candidates = sorted((ROOT_DIR / "models" / "trained").glob("**/*best*.pt"))
    if candidates:
        return candidates[-1]
    return ROOT_DIR / "models" / "checkpoints" / "best.pt"


def _default_source() -> str:
    test_images = ROOT_DIR / "data" / "processed" / "steel-surface-defect" / "test" / "images"
    if test_images.exists():
        first = next(iter(sorted(test_images.glob("*"))), None)
        if first is not None:
            return str(first)
    return "0"


def _default_results_csv() -> Path:
    candidates = sorted((ROOT_DIR / "runs").glob("**/results.csv"))
    if candidates:
        return candidates[-1]
    return ROOT_DIR / "runs" / "detect" / "train" / "results.csv"


def _load_stylesheet() -> str:
    qss = ROOT_DIR / "apps" / "desktop" / "style.qss"
    if qss.exists():
        return qss.read_text(encoding="utf-8")
    return ""


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(_load_stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
