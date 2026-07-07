import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from od_platform.frame_source import (
    CameraConfig,
    FrameSource,
    SourceType,
    create_frame_source,
    detect_source_type,
)


class TestFrameSource(unittest.TestCase):
    def _write_image(self, path: Path, value: int = 127) -> None:
        image = np.full((8, 12, 3), value, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))

    def test_camera_config_validates_and_returns_resolution(self) -> None:
        config = CameraConfig(camera_id=1, width=640, height=480, fps=60, backend="msmf", codec="MJPG")

        self.assertEqual(config.get_resolution(), (640, 480, 60))

        with self.assertRaises(ValueError):
            CameraConfig(width=0)

    def test_detect_source_type_from_camera_image_video_and_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "a.jpg"
            video = root / "a.mp4"
            image.write_bytes(b"not-used")
            video.write_bytes(b"not-used")

            self.assertEqual(detect_source_type(0), SourceType.CAMERA)
            self.assertEqual(detect_source_type("0"), SourceType.CAMERA)
            self.assertEqual(detect_source_type(image), SourceType.IMAGE)
            self.assertEqual(detect_source_type(video), SourceType.VIDEO)
            self.assertEqual(detect_source_type(root), SourceType.IMAGE_FOLDER)

    def test_image_frame_source_reads_once_and_exposes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.png"
            self._write_image(path)

            with create_frame_source(path) as source:
                frame = source.read()
                self.assertIsNotNone(frame)
                self.assertEqual(frame.info.source_type, SourceType.IMAGE)
                self.assertEqual(frame.resolution, (12, 8))
                self.assertEqual(frame.info.filename, "sample.png")
                self.assertIsNone(source.read())

    def test_image_folder_source_reads_sorted_images_with_stride(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_image(root / "b.jpg", 10)
            self._write_image(root / "a.jpg", 20)
            self._write_image(root / "c.jpg", 30)

            with create_frame_source(root) as source:
                source.set_stride(2)
                first = source.read()
                second = source.read()
                third = source.read()

            self.assertEqual(first.info.filename, "a.jpg")
            self.assertEqual(second.info.filename, "c.jpg")
            self.assertIsNone(third)

    def test_frame_source_context_raises_when_open_fails(self) -> None:
        class BrokenSource(FrameSource):
            def open(self) -> bool:
                return False

            def read(self):
                return None

            def close(self) -> None:
                pass

            def get_source_type(self) -> SourceType:
                return SourceType.IMAGE

        with self.assertRaises(RuntimeError):
            with BrokenSource("missing"):
                pass


if __name__ == "__main__":
    unittest.main()
