import asyncio
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from od_platform.frame_source import (
    CameraConfig,
    CameraSource,
    FrameSource,
    ImageSource,
    SourceType,
    ThreadedSource,
    VideoSource,
    create_async_source,
    create_frame_source,
    create_threaded_source,
    detect_source_type,
)
from od_platform.frame_source.sources import (
    CameraSource as SourcesCameraSource,
)
from od_platform.frame_source.sources import (
    ImageFolderSource as SourcesImageFolderSource,
)
from od_platform.frame_source.sources import (
    ImageSource as SourcesImageSource,
)
from od_platform.frame_source.sources import (
    VideoSource as SourcesVideoSource,
)


class TestFrameSource(unittest.TestCase):
    def _write_image(self, path: Path, value: int = 127) -> None:
        image = np.full((8, 12, 3), value, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))

    def _write_video(self, path: Path) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (12, 8))
        self.assertTrue(writer.isOpened())
        for value in (20, 40, 60):
            writer.write(np.full((8, 12, 3), value, dtype=np.uint8))
        writer.release()

    def test_camera_config_validates_and_returns_resolution(self) -> None:
        config = CameraConfig(camera_id=1, width=640, height=480, fps=60, backend="msmf", codec="MJPG")

        self.assertEqual(config.get_resolution(), (640, 480, 60))

        with self.assertRaises(ValueError):
            CameraConfig(width=0)

    def test_camera_source_forces_stride_to_one(self) -> None:
        source = CameraSource(CameraConfig())

        source.set_stride(3)

        self.assertEqual(source.stride(), 1)
        self.assertEqual(source.get_source_type(), SourceType.CAMERA)

    def test_detect_source_type_from_camera_image_video_and_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "a.jpg"
            video = root / "a.mp4"
            image.write_bytes(b"not-used")
            video.write_bytes(b"not-used")

            self.assertEqual(detect_source_type(0), SourceType.CAMERA)
            self.assertEqual(detect_source_type("0"), SourceType.CAMERA)
            self.assertEqual(detect_source_type("rtsp://127.0.0.1/live"), SourceType.VIDEO)
            self.assertEqual(detect_source_type("http://example.local/live.m3u8"), SourceType.VIDEO)
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

    def test_image_source_name_reads_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.jpg"
            self._write_image(path)

            with ImageSource(path) as source:
                frame = source.read()

            self.assertIsNotNone(frame)
            self.assertEqual(frame.info.source_path, str(path))

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

    def test_factory_sets_stride_on_video_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.avi"
            self._write_video(path)

            with create_frame_source(path, stride=2) as source:
                indexes = [frame.info.frame_index for frame in source]

            self.assertEqual(indexes, [0, 2])

    def test_video_source_reads_metadata_and_supports_seek(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.avi"
            self._write_video(path)

            with VideoSource(path) as source:
                frame = source.read()
                self.assertTrue(source.seek(frame=1))
                second = source.read()

            self.assertIsNotNone(frame)
            self.assertIsNotNone(second)
            self.assertEqual(frame.info.source_type, SourceType.VIDEO)
            self.assertEqual(frame.info.filename, "sample.avi")
            self.assertEqual(frame.resolution, (12, 8))

    def test_threaded_source_reads_bounded_folder_without_dropping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_image(root / "a.jpg", 10)
            self._write_image(root / "b.jpg", 20)
            self._write_image(root / "c.jpg", 30)

            with create_threaded_source(root, buffer="bounded", buffer_size=8) as source:
                frames = list(source)

            self.assertIsInstance(source, ThreadedSource)
            self.assertEqual([frame.info.filename for frame in frames], ["a.jpg", "b.jpg", "c.jpg"])

    def test_async_source_iterates_without_threading(self) -> None:
        async def collect(path: Path) -> list[int]:
            indexes: list[int] = []
            async with create_async_source(path, stride=2, threaded=False) as source:
                async for frame in source:
                    indexes.append(frame.info.frame_index)
            return indexes

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.avi"
            self._write_video(path)

            indexes = asyncio.run(collect(path))

        self.assertEqual(indexes, [0, 2])

    def test_sources_package_exports_concrete_sources(self) -> None:
        self.assertIs(SourcesCameraSource, CameraSource)
        self.assertIs(SourcesImageSource, ImageSource)
        self.assertEqual(SourcesImageFolderSource.__name__, "ImageFolderSource")
        self.assertIs(SourcesVideoSource, VideoSource)

    def test_factory_rejects_missing_local_path(self) -> None:
        with self.assertRaises(ValueError):
            create_frame_source("missing.mp4")

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
