"""ModelCatalogView 离屏冒烟：刷新/信号/一键应用。"""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DESKTOP_DIR = Path(__file__).resolve().parents[4] / "desktop"
if str(DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(DESKTOP_DIR))

from PySide6.QtWidgets import QApplication  # noqa: E402

from views.model_catalog_view import ModelCatalogView  # noqa: E402

_app = QApplication.instance() or QApplication([])


class TestModelCatalogView(unittest.TestCase):
    def setUp(self) -> None:
        self.view = ModelCatalogView()

    def test_refresh_populates_models(self) -> None:
        self.assertGreaterEqual(self.view.model_list.count(), 30)

    def test_family_filter(self) -> None:
        self.view.family_combo.setCurrentText("yolov8")
        self.assertEqual(self.view.model_list.count(), 5)

    def test_families_changed_emitted(self) -> None:
        received: list[list] = []
        self.view.families_changed.connect(received.append)
        self.view.refresh()
        self.assertTrue(received)
        self.assertIn("yolov8", received[-1])

    def test_recommend_filters(self) -> None:
        self.view.recommend_edit.setText("最准")
        self.view.recommend_button.click()
        self.assertGreaterEqual(self.view.model_list.count(), 1)

    def test_apply_to_train_emits_signal(self) -> None:
        applied: list[str] = []
        self.view.applied_to_train.connect(applied.append)
        self.view.model_list.setCurrentRow(0)
        self.view.apply_train_button.click()
        self.assertEqual(len(applied), 1)
        self.assertTrue(applied[0].endswith(".pt"))

    def test_apply_without_selection_emits_hint(self) -> None:
        statuses: list[tuple[str, str]] = []
        self.view.status_changed.connect(lambda s, d: statuses.append((s, d)))
        self.view.model_list.clear()
        self.view.apply_train_button.click()
        self.assertEqual(statuses[0][0], "提示")


if __name__ == "__main__":
    unittest.main()
