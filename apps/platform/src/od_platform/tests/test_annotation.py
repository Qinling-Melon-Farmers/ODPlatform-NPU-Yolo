import tempfile
import unittest
from pathlib import Path

from od_platform.annotation.canvas import (
    _imread_unicode,
    boxes_to_display_coords,
    compute_display_scale,
)
from od_platform.annotation.session import AnnotationSession
from od_platform.annotation.writer import BBox, bbox_from_pixels, read_yolo_label, write_yolo_label


class TestYoloWriter(unittest.TestCase):
    def test_bbox_from_pixels_normalized(self) -> None:
        box = bbox_from_pixels(2, 10, 20, 30, 60, 100, 100)
        self.assertEqual(box.class_id, 2)
        self.assertAlmostEqual(box.x_center, 0.2)
        self.assertAlmostEqual(box.y_center, 0.4)
        self.assertAlmostEqual(box.width, 0.2)
        self.assertAlmostEqual(box.height, 0.4)

    def test_bbox_pixels_roundtrip(self) -> None:
        box = bbox_from_pixels(0, 25, 25, 75, 75, 100, 100)
        x1, y1, x2, y2 = box.to_pixels(100, 100)
        self.assertEqual((x1, y1, x2, y2), (25, 25, 75, 75))

    def test_bbox_clamps_out_of_bounds(self) -> None:
        box = bbox_from_pixels(0, -50, -50, 200, 200, 100, 100)
        # 中心 (75, 75) 落在图像内不裁剪；宽高 250px 被裁剪到 1.0
        self.assertEqual(box.x_center, 0.75)
        self.assertEqual(box.y_center, 0.75)
        self.assertEqual(box.width, 1.0)
        self.assertEqual(box.height, 1.0)

    def test_bbox_normalizes_reversed_corners(self) -> None:
        box = bbox_from_pixels(0, 30, 60, 10, 20, 100, 100)
        self.assertAlmostEqual(box.x_center, 0.2)
        self.assertAlmostEqual(box.y_center, 0.4)

    def test_write_read_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            label_path = Path(temp_dir) / "img001.txt"
            boxes = [
                BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.3),
                BBox(class_id=1, x_center=0.25, y_center=0.75, width=0.1, height=0.1),
            ]
            write_yolo_label(label_path, boxes)
            loaded = read_yolo_label(label_path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0], boxes[0])
            self.assertEqual(loaded[1], boxes[1])

    def test_write_skips_invalid_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            label_path = Path(temp_dir) / "img001.txt"
            boxes = [
                BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.0, height=0.3),
                BBox(class_id=1, x_center=0.5, y_center=0.5, width=0.2, height=0.2),
            ]
            write_yolo_label(label_path, boxes)
            loaded = read_yolo_label(label_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].class_id, 1)

    def test_read_missing_file_returns_empty(self) -> None:
        self.assertEqual(read_yolo_label(Path("no-such-file.txt")), [])

    def test_read_ignores_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            label_path = Path(temp_dir) / "img001.txt"
            label_path.write_text(
                "0 0.5 0.5 0.2 0.2\nnot a label line\n1 0.5\nabc 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            loaded = read_yolo_label(label_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].class_id, 0)


class TestAnnotationSession(unittest.TestCase):
    def _make_dataset(self, root: Path, image_names: list[str]) -> tuple[Path, Path]:
        images_dir = root / "images"
        labels_dir = root / "annotations"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        for name in image_names:
            (images_dir / name).write_bytes(b"image")
        # 混入非图片文件，应被忽略
        (images_dir / "notes.txt").write_text("x", encoding="utf-8")
        return images_dir, labels_dir

    def test_session_progress_and_next_unannotated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir, labels_dir = self._make_dataset(Path(temp_dir), ["a.jpg", "b.png", "c.jpg"])
            session = AnnotationSession(images_dir=images_dir, labels_dir=labels_dir, classes=["cat", "dog"])

            self.assertEqual(session.total_images, 3)
            self.assertEqual(session.progress, (0, 3))
            self.assertEqual(session.next_unannotated().name, "a.jpg")

    def test_session_save_updates_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir, labels_dir = self._make_dataset(Path(temp_dir), ["a.jpg", "b.jpg"])
            session = AnnotationSession(images_dir=images_dir, labels_dir=labels_dir, classes=["cat"])

            box = bbox_from_pixels(0, 10, 10, 50, 50, 100, 100)
            path = session.save_labels("a", [box])
            self.assertTrue(path.exists())
            self.assertEqual(session.progress, (1, 2))
            self.assertEqual(session.next_unannotated().name, "b.jpg")

    def test_session_resume_discovers_annotated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir, labels_dir = self._make_dataset(Path(temp_dir), ["a.jpg", "b.jpg"])
            first = AnnotationSession(images_dir=images_dir, labels_dir=labels_dir, classes=["cat"])
            first.save_labels("a", [bbox_from_pixels(0, 10, 10, 50, 50, 100, 100)])

            resumed = AnnotationSession(images_dir=images_dir, labels_dir=labels_dir, classes=["cat"])
            self.assertEqual(resumed.progress, (1, 2))
            self.assertEqual(resumed.next_unannotated().name, "b.jpg")

    def test_session_skips_out_of_range_class_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir, labels_dir = self._make_dataset(Path(temp_dir), ["a.jpg"])
            session = AnnotationSession(images_dir=images_dir, labels_dir=labels_dir, classes=["cat"])
            path = session.save_labels(
                "a",
                [BBox(class_id=5, x_center=0.5, y_center=0.5, width=0.2, height=0.2)],
            )
            loaded = read_yolo_label(path)
            self.assertEqual(loaded, [])

    def test_session_empty_labels_still_marks_annotated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir, labels_dir = self._make_dataset(Path(temp_dir), ["bg.jpg"])
            session = AnnotationSession(images_dir=images_dir, labels_dir=labels_dir, classes=["cat"])
            session.save_labels("bg", [])
            self.assertEqual(session.progress, (1, 1))
            self.assertIsNone(session.next_unannotated())


class TestCanvasImageIO(unittest.TestCase):
    def test_imread_unicode_path(self) -> None:
        """cv2.imread 无法读取含中文路径，_imread_unicode 必须可读（回归测试）。"""
        try:
            import cv2
            import numpy as np
        except ImportError:  # pragma: no cover
            self.skipTest("opencv-python 不可用")
        with tempfile.TemporaryDirectory() as temp_dir:
            # 中文目录名复现 Windows 非 ASCII 路径问题
            image_path = Path(temp_dir) / "中文标注测试" / "样例.jpg"
            image_path.parent.mkdir(parents=True)
            canvas = np.zeros((64, 96, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", canvas)
            self.assertTrue(ok)
            encoded.tofile(str(image_path))

            loaded = _imread_unicode(image_path)
            self.assertIsNotNone(loaded)
            if loaded is not None:
                self.assertEqual(loaded.shape, (64, 96, 3))

    def test_imread_unicode_missing_file(self) -> None:
        self.assertIsNone(_imread_unicode(Path("不存在/缺失.jpg")))


class TestCanvasMath(unittest.TestCase):
    def test_compute_display_scale_large_image(self) -> None:
        self.assertAlmostEqual(compute_display_scale(2000, 1000, 1200), 0.6)

    def test_compute_display_scale_small_image(self) -> None:
        self.assertEqual(compute_display_scale(100, 50, 1200), 1.0)

    def test_compute_display_scale_zero_size(self) -> None:
        self.assertEqual(compute_display_scale(0, 100, 1200), 1.0)

    def test_boxes_to_display_coords_scales_down(self) -> None:
        """编辑模式：原图框应缩放到显示坐标（scale<1 时），避免双重缩放。"""
        box = BBox(class_id=1, x_center=0.5, y_center=0.5, width=0.2, height=0.2)
        coords = boxes_to_display_coords([box], image_size=(100, 100), scale=0.5)
        # 原图像素 (40,40,60,60) → 显示坐标 (20,20,30,30)
        self.assertEqual(coords, [(20, 20, 30, 30, 1)])

    def test_boxes_to_display_roundtrip_preserves_geometry(self) -> None:
        """显示坐标转回归一化后应与原框一致（编辑-保存往返不漂移）。"""
        box = BBox(class_id=0, x_center=0.3, y_center=0.6, width=0.4, height=0.2)
        scale = 0.6
        (x1, y1, x2, y2, class_id) = boxes_to_display_coords([box], image_size=(200, 100), scale=scale)[0]
        restored = bbox_from_pixels(class_id, x1 / scale, y1 / scale, x2 / scale, y2 / scale, 200, 100)
        self.assertAlmostEqual(restored.x_center, box.x_center, places=2)
        self.assertAlmostEqual(restored.y_center, box.y_center, places=2)
        self.assertAlmostEqual(restored.width, box.width, places=2)
        self.assertAlmostEqual(restored.height, box.height, places=2)

    def test_boxes_to_display_coords_identity_scale(self) -> None:
        box = BBox(class_id=2, x_center=0.5, y_center=0.5, width=0.5, height=0.5)
        coords = boxes_to_display_coords([box], image_size=(100, 100), scale=1.0)
        self.assertEqual(coords, [(25, 25, 75, 75, 2)])


if __name__ == "__main__":
    unittest.main()
