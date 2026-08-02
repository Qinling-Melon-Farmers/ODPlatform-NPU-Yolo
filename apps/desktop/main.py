"""PySide6 desktop workbench for ODPlatform Studio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PLATFORM_SRC = ROOT_DIR / "apps" / "platform" / "src"

try:
    import cv2
    from PySide6.QtCore import Qt, QThread, Slot
    from PySide6.QtGui import QImage, QPixmap
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

from infer_worker import InferWorker  # noqa: E402

# 兼容别名（Step 7 清理；新代码请从 services.* 导入）
from services import file_utils  # noqa: E402
from services import paths as desktop_paths  # noqa: E402
from task_worker import CommandWorker  # noqa: E402

_read_json = file_utils.read_json
_read_csv_rows = file_utils.read_csv_rows
_format_mapping = file_utils.format_mapping
_filter_paths = file_utils.filter_paths
_split_extra_args = file_utils.split_extra_args
_open_path = file_utils.open_path
_open_selected_parent = file_utils.open_selected_parent
_is_relative_to = desktop_paths.is_relative_to
_default_model = desktop_paths.default_model
_default_source = desktop_paths.default_source
_default_results_csv = desktop_paths.default_results_csv


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
        self._build_layout()
        self._connect_signals()
        self._refresh_all_result_tabs()

    def _sync_agent_config_fields(self) -> None:
        """AI 助手页（AgentChatView）与 AI 任务页（TaskLauncherView）配置双向同步。"""
        self._syncing_agent_config = False
        view = self.agent_view

        # view → 任务页
        view.config_changed.connect(self._apply_view_config_to_task_page)
        # 任务页 → view（经 TaskLauncherView.agent_config_changed）
        self.task_launcher_view.agent_config_changed.connect(self._apply_task_config_to_view)

    @Slot(str, str, str)
    def _apply_view_config_to_task_page(self, base_url: str, model: str, api_key: str) -> None:
        if self._syncing_agent_config:
            return
        self._syncing_agent_config = True
        self.task_launcher_view.set_agent_config(base_url, model, api_key)
        self._syncing_agent_config = False

    @Slot(str, str, str)
    def _apply_task_config_to_view(self, base_url: str, model: str, api_key: str) -> None:
        if self._syncing_agent_config:
            return
        self._syncing_agent_config = True
        self.agent_view.set_config(base_url, model, api_key)
        self._syncing_agent_config = False

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name.
        self._stop_worker()
        self.task_launcher_view.shutdown()
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
        agent_view = self._build_agent_tab()
        self.agent_view = agent_view
        tabs.addTab(agent_view, "AI 助手")
        model_catalog_view = self._build_model_catalog_tab()
        self.model_catalog_view = model_catalog_view
        tabs.addTab(model_catalog_view, "模型目录")
        training_view = self._build_training_tab()
        self.training_view = training_view
        tabs.addTab(training_view, "训练结果")
        tabs.addTab(self._build_annotation_review_tab(), "标注复核")
        task_launcher_view = self._build_task_launcher_tab()
        self.task_launcher_view = task_launcher_view
        tabs.addTab(task_launcher_view, "任务启动")
        root.addWidget(tabs)
        self.setCentralWidget(central)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("Header")
        layout = QHBoxLayout(header)
        logo = QLabel("OD")
        logo.setObjectName("AppLogo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("ODPlatform Studio")
        title.setObjectName("AppTitle")
        subtitle = QLabel("目标检测开发平台 | 推理、评估、质检、训练、AI 助手、任务启动")
        subtitle.setObjectName("AppSubtitle")
        text = QVBoxLayout()
        text.addWidget(title)
        text.addWidget(subtitle)
        layout.addWidget(logo)
        layout.addSpacing(12)
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
        from views.agent_chat_view import AgentChatView

        return AgentChatView()

    def _build_annotation_review_tab(self) -> QWidget:
        from views.annotation_review import AnnotationReviewView

        return AnnotationReviewView()

    def _build_task_launcher_tab(self) -> QWidget:
        from views.task_launcher_view import TaskLauncherView

        return TaskLauncherView()

    def _build_model_catalog_tab(self) -> QWidget:
        from views.model_catalog_view import ModelCatalogView

        return ModelCatalogView()

    def _build_training_tab(self) -> QWidget:
        from views.training_result_view import TrainingResultView

        return TrainingResultView()

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self._start_worker)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop_worker)
        self.open_output_button.clicked.connect(self._open_last_output_dir)
        self.eval_list.currentItemChanged.connect(lambda current, _previous: self._show_eval_item(current))
        self.validation_list.currentItemChanged.connect(lambda current, _previous: self._show_validation_item(current))
        # 模型目录视图跨视图连接
        self.model_catalog_view.applied_to_train.connect(self._on_model_applied_to_train)
        self.model_catalog_view.applied_to_infer.connect(self._on_model_applied_to_infer)
        self.model_catalog_view.families_changed.connect(self.task_launcher_view.set_model_families)
        self.model_catalog_view.status_changed.connect(self._set_status)
        self.eval_filter_edit.textChanged.connect(lambda _text: self._refresh_evaluation_results())
        self.validation_filter_edit.textChanged.connect(lambda _text: self._refresh_validation_reports())
        # 任务启动页（TaskLauncherView）跨视图连接
        self.task_launcher_view.status_changed.connect(self._set_status)
        self.task_launcher_view.task_finished.connect(self._refresh_all_result_tabs)
        self.task_launcher_view.agent_config_changed.connect(self._apply_task_config_to_view)
        # AI 助手页与 AI 任务页共享 API 配置（双向同步，防重复填写）
        self._sync_agent_config_fields()
        # AI 助手状态转发到主窗口状态栏
        self.agent_view.status_changed.connect(self._set_status)
        self.model_catalog_view.refresh()

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

    def _task_tab_index(self) -> int:
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == "任务启动":
                return index
        return self.tabs.count() - 1  # 兜底：最后一个 tab

    def _open_model_catalog(self) -> None:
        """跳转到模型目录标签页并刷新。"""
        self.tabs.setCurrentIndex(self._model_catalog_tab_index())
        self.model_catalog_view.refresh()

    @Slot(str)
    def _on_model_applied_to_train(self, name: str) -> None:
        """模型目录一键应用：回填训练页并切换。"""
        self.task_launcher_view.set_train_model(name)
        self.task_launcher_view.select_task("模型训练")
        self.tabs.setCurrentIndex(self._task_tab_index())
        self._set_status("就绪", f"已应用到模型训练: {name}")

    @Slot(str)
    def _on_model_applied_to_infer(self, name: str) -> None:
        """模型目录一键应用：回填推理页并切换。"""
        self.model_edit.setText(name)
        self.tabs.setCurrentIndex(0)
        self._set_status("就绪", f"已应用到模型推理: {name}")

    def _model_catalog_tab_index(self) -> int:
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == "模型目录":
                return index
        return 3  # 兜底：推理/评估/质检之后

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", str(ROOT_DIR), "PyTorch weights (*.pt);;All files (*)")
        if path:
            self.model_edit.setText(path)

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
        self.training_view.refresh()

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

    def _open_selected_eval(self) -> None:
        _open_selected_parent(self.eval_list)

    def _open_selected_validation(self) -> None:
        _open_selected_parent(self.validation_list)

    def _set_status(self, state: str, detail: str) -> None:
        self.status_label.setText(f"状态：{state} | {detail}")

    def _append_log(self, message: str) -> None:
        self.infer_log.append(message)










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
