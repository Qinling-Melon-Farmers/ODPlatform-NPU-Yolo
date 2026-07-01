import json
import tempfile
import unittest
from pathlib import Path

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, list_capabilities
from od_platform.data_pipeline.convert.service import convert_data_to_yolo


class TestDataPipelineConvert(unittest.TestCase):
    def test_registry_lists_detect_converters(self) -> None:
        capabilities = list_capabilities()

        self.assertIn(AnnotationFormat.PASCAL_VOC, capabilities)
        self.assertIn(AnnotationFormat.COCO, capabilities)
        self.assertIn(AnnotationFormat.YOLO, capabilities)
        self.assertIn(Task.DETECT, capabilities[AnnotationFormat.COCO])

    def test_coco_json_converts_to_yolo_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            coco_path = root / "instances.json"
            output_dir = root / "labels"
            coco_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 50},
                            {"id": 2, "file_name": "b.jpg", "width": 40, "height": 40},
                        ],
                        "categories": [
                            {"id": 3, "name": "ship"},
                            {"id": 8, "name": "aircraft"},
                        ],
                        "annotations": [
                            {
                                "id": 10,
                                "image_id": 1,
                                "category_id": 3,
                                "bbox": [10, 5, 20, 10],
                                "iscrowd": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            classes = convert_data_to_yolo(
                input_dir=coco_path,
                output_labels_dir=output_dir,
                annotation_format=AnnotationFormat.COCO,
                options=ConvertOptions(task=Task.DETECT),
            )

            self.assertEqual(classes, ["ship", "aircraft"])
            self.assertEqual(
                (output_dir / "a.txt").read_text(encoding="utf-8"),
                "0 0.200000 0.200000 0.200000 0.200000",
            )
            self.assertEqual((output_dir / "b.txt").read_text(encoding="utf-8"), "")

    def test_yolo_converter_validates_and_normalizes_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "image_1.txt").write_text("1 0.5 0.25 0.2 0.1\n", encoding="utf-8")

            classes = convert_data_to_yolo(
                input_dir=input_dir,
                output_labels_dir=output_dir,
                annotation_format=AnnotationFormat.YOLO,
                options=ConvertOptions(task=Task.DETECT, classes=["ship", "aircraft"]),
            )

            self.assertEqual(classes, ["ship", "aircraft"])
            self.assertEqual(
                (output_dir / "image_1.txt").read_text(encoding="utf-8"),
                "1 0.500000 0.250000 0.200000 0.100000",
            )

    def test_yolo_converter_rejects_invalid_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "bad.txt").write_text("0 1.2 0.5 0.2 0.1\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                convert_data_to_yolo(
                    input_dir=input_dir,
                    output_labels_dir=output_dir,
                    annotation_format=AnnotationFormat.YOLO,
                    options=ConvertOptions(task=Task.DETECT),
                )

    def test_service_rejects_unsupported_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "labels"
            input_dir.mkdir()
            (input_dir / "a.txt").write_text("", encoding="utf-8")

            with self.assertRaises(ValueError):
                convert_data_to_yolo(
                    input_dir=input_dir,
                    output_labels_dir=root / "out",
                    annotation_format=AnnotationFormat.YOLO,
                    options=ConvertOptions(task=Task.SEGMENT),
                )


if __name__ == "__main__":
    unittest.main()
