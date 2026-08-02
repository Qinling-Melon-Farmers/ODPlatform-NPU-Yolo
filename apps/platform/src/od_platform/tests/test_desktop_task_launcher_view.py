"""TaskLauncherView 离屏冒烟：预览/脱敏/参数收集/启动委托。"""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DESKTOP_DIR = Path(__file__).resolve().parents[4] / "desktop"
if str(DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(DESKTOP_DIR))

from PySide6.QtWidgets import QApplication  # noqa: E402
from views.task_launcher_view import TaskLauncherView  # noqa: E402

_app = QApplication.instance() or QApplication([])


class TestTaskLauncherView(unittest.TestCase):
    def setUp(self) -> None:
        self.view = TaskLauncherView()
        self.view._controller = None  # 断开真实控制器（避免误启动）

    def test_task_names_populated(self) -> None:
        self.assertEqual(self.view.desktop_task_combo.count(), 11)

    def test_preview_list_models(self) -> None:
        self.view.desktop_task_combo.setCurrentText("列出模型")
        self.view.list_models_recommend_edit.setText("最准")
        preview = self.view.task_command_preview.toPlainText()
        self.assertIn("model_catalog.cli.list_models", preview)
        self.assertIn("--recommend", preview)

    def test_preview_masks_api_key(self) -> None:
        self.view.desktop_task_combo.setCurrentText("AI 任务")
        self.view.ai_task_prompt_edit.setText("训练 demo")
        self.view.ai_task_base_url_edit.setText("https://x/v1")
        self.view.ai_task_model_edit.setText("deepseek")
        self.view.ai_task_api_key_edit.setText("sk-secret-123")
        preview = self.view.task_command_preview.toPlainText()
        self.assertNotIn("sk-secret-123", preview)
        self.assertIn("***", preview)

    def test_preview_incomplete_shows_hint(self) -> None:
        self.view.desktop_task_combo.setCurrentText("导入数据集")
        self.view.import_zip_edit.setText("")
        preview = self.view.task_command_preview.toPlainText()
        self.assertIn("参数待补全", preview)

    def test_collect_params_train(self) -> None:
        self.view.desktop_task_combo.setCurrentText("模型训练")
        self.view.train_model_edit.setText("yolo11n.pt")
        self.view.train_dataset_edit.setText("demo")
        self.view.train_epochs_spin.setValue(50)
        self.view.train_batch_spin.setValue(16)
        params = self.view._collect_task_params()
        self.assertEqual(params["model"], "yolo11n.pt")
        self.assertEqual(params["epochs"], 50)

    def test_start_delegates_to_controller(self) -> None:
        calls: list[tuple] = []

        class FakeController:
            def start(self, module, args) -> None:
                calls.append((module, args))

            def stop(self) -> None:
                pass

        self.view._controller = FakeController()
        self.view.desktop_task_combo.setCurrentText("列出模型")
        self.view.list_models_recommend_edit.setText("最快")
        self.view._on_start_clicked()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "od_platform.model_catalog.cli.list_models")

    def test_start_value_error_appends_output(self) -> None:
        class FakeController:
            def start(self, module, args) -> None:
                pass

            def stop(self) -> None:
                pass

        self.view._controller = FakeController()
        self.view.desktop_task_combo.setCurrentText("导入数据集")
        self.view.import_zip_edit.setText("")
        self.view._on_start_clicked()
        self.assertIn("参数错误", self.view.task_output.toPlainText())


if __name__ == "__main__":
    unittest.main()
