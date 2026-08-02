import csv
import sys
import tempfile
import unittest
from pathlib import Path

DESKTOP_DIR = Path(__file__).resolve().parents[4] / "desktop"
if str(DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(DESKTOP_DIR))

from views.annotation_review_view import (  # noqa: E402
    boxes_to_rows,
    draw_boxes,
    find_image_path,
    list_audit_runs,
    load_review_queue,
    rows_to_boxes,
)

from od_platform.annotation.writer import BBox  # noqa: E402


class TestReviewQueueParsing(unittest.TestCase):
    def test_list_audit_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotation_root = root / "annotation"
            (annotation_root / "20260802_100000").mkdir(parents=True)
            (annotation_root / "20260802_090000").mkdir()
            runs = list_audit_runs(root)
            self.assertEqual([run.name for run in runs], ["20260802_100000", "20260802_090000"])

    def test_list_audit_runs_missing_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(list_audit_runs(Path(temp_dir)), [])

    def test_load_review_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit_dir = Path(temp_dir) / "run"
            audit_dir.mkdir()
            with (audit_dir / "review_queue.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image", "boxes", "discarded", "reason", "label_path"],
                )
                writer.writeheader()
                writer.writerow({"image": "a.jpg", "boxes": "0", "discarded": "0", "reason": "empty", "label_path": "/x/a.txt"})
            rows = load_review_queue(audit_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["image"], "a.jpg")
            self.assertEqual(rows[0]["reason"], "empty")

    def test_load_review_queue_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(load_review_queue(Path(temp_dir) / "none"), [])


class TestImagePathResolution(unittest.TestCase):
    def test_find_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            images_dir.mkdir(parents=True)
            (images_dir / "img001.jpg").write_bytes(b"x")
            label_path = root / "annotations" / "img001.txt"
            result = find_image_path(label_path, "img001.jpg")
            self.assertEqual(result, images_dir / "img001.jpg")

    def test_find_image_path_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label_path = root / "annotations" / "img001.txt"
            self.assertIsNone(find_image_path(label_path, "img001.jpg"))


class TestBoxTableRoundtrip(unittest.TestCase):
    def test_boxes_to_rows_and_back(self) -> None:
        boxes = [BBox(class_id=1, x_center=0.5, y_center=0.4, width=0.2, height=0.3)]
        rows = boxes_to_rows(boxes)
        self.assertEqual(rows[0][0], "1")
        restored = rows_to_boxes(rows)
        self.assertEqual(len(restored), 1)
        self.assertAlmostEqual(restored[0].x_center, 0.5)

    def test_rows_to_boxes_skips_invalid(self) -> None:
        rows = [
            ["0", "0.5", "0.5", "0.2", "0.2"],
            ["x", "0.5", "0.5", "0.2", "0.2"],  # 非法类别
            ["1", "0.5", "0.5", "0", "0.2"],  # 零宽
        ]
        boxes = rows_to_boxes(rows)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].class_id, 0)

    def test_draw_boxes_returns_copy_with_boxes(self) -> None:
        import numpy as np

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.2)]
        frame = draw_boxes(image, boxes)
        self.assertEqual(frame.shape, image.shape)
        self.assertFalse((frame == image).all())  # 有绘制内容


if __name__ == "__main__":
    unittest.main()
