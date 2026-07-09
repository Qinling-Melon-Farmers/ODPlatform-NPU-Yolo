import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from od_platform.common import paths
from od_platform.common.constants import AnnotationFormat, SplitStrategy
from od_platform.data_pipeline.orchestrator import DatasetPipeline


class TestDatasetPipeline(unittest.TestCase):
    def _write_voc_sample(self, root: Path, index: int, class_name: str) -> None:
        image = root / "raw" / "demo" / "images" / f"sample_{index:03d}.jpg"
        xml = root / "raw" / "demo" / "annotations" / f"sample_{index:03d}.xml"
        image.parent.mkdir(parents=True, exist_ok=True)
        xml.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-{index}".encode())
        xml.write_text(
            f"""<annotation>
  <filename>{image.name}</filename>
  <size><width>100</width><height>100</height><depth>3</depth></size>
  <object>
    <name>{class_name}</name>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>60</ymax></bndbox>
  </object>
</annotation>
""",
            encoding="utf-8",
        )

    def _write_yolo_sample(self, root: Path, index: int, class_id: int) -> None:
        image = root / "raw" / "demo-yolo" / "images" / f"sample_{index:03d}.jpg"
        label = root / "raw" / "demo-yolo" / "annotations" / f"sample_{index:03d}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-{index}".encode())
        label.write_text(f"{class_id} 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")

    def test_pipeline_converts_splits_materializes_and_writes_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, class_name in enumerate(["cat", "cat", "dog", "dog"]):
                self._write_voc_sample(root, index, class_name)

            processed_dir = root / "processed"
            configs_dir = root / "configs" / "datasets"
            with (
                patch.object(paths, "RAW_DATA_DIR", root / "raw"),
                patch.object(paths, "PROCESSED_DATA_DIR", processed_dir),
                patch.object(paths, "DATASET_CONFIGS_DIR", configs_dir),
            ):
                result = DatasetPipeline(
                    "demo",
                    AnnotationFormat.PASCAL_VOC,
                    train_rate=0.5,
                    val_rate=0.25,
                    split_strategy=SplitStrategy.RANDOM,
                    random_state=42,
                ).run()

            self.assertEqual(result["counts"], {"train": 2, "val": 1, "test": 1})
            self.assertTrue((processed_dir / "demo" / "train" / "images").is_dir())
            self.assertTrue((processed_dir / "demo" / "val" / "labels").is_dir())
            yaml_path = configs_dir / "demo.yaml"
            self.assertEqual(result["yaml"], str(yaml_path))
            content = yaml_path.read_text(encoding="utf-8")
            self.assertIn("odp_meta:", content)
            self.assertIn("dataset_name: demo", content)
            self.assertIn("source_format: pascal_voc", content)

    def test_pipeline_reads_yolo_classes_from_raw_data_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_root = root / "raw" / "demo-yolo"
            raw_root.mkdir(parents=True)
            (raw_root / "data.yaml").write_text(
                "nc: 2\nnames:\n  0: bridge\n  1: tank\n",
                encoding="utf-8",
            )
            for index, class_id in enumerate([0, 0, 1, 1]):
                self._write_yolo_sample(root, index, class_id)

            processed_dir = root / "processed"
            configs_dir = root / "configs" / "datasets"
            with (
                patch.object(paths, "RAW_DATA_DIR", root / "raw"),
                patch.object(paths, "PROCESSED_DATA_DIR", processed_dir),
                patch.object(paths, "DATASET_CONFIGS_DIR", configs_dir),
            ):
                result = DatasetPipeline(
                    "demo-yolo",
                    AnnotationFormat.YOLO,
                    train_rate=0.5,
                    val_rate=0.25,
                    split_strategy=SplitStrategy.RANDOM,
                    random_state=42,
                ).run()

            self.assertEqual(result["counts"], {"train": 2, "val": 1, "test": 1})
            content = (configs_dir / "demo-yolo.yaml").read_text(encoding="utf-8")
            self.assertIn("0: bridge", content)
            self.assertIn("1: tank", content)
            self.assertIn("source_format: yolo", content)


if __name__ == "__main__":
    unittest.main()
