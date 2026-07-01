import tempfile
import unittest
from pathlib import Path

from od_platform.common.constants import AnnotationFormat, SplitStrategy, Task
from od_platform.data_pipeline.split.manifest import SplitManifest
from od_platform.data_pipeline.split.materializer import SplitOutputDirs
from od_platform.data_pipeline.split.yaml_writer import write_dataset_yaml


class TestYamlWriter(unittest.TestCase):
    def test_write_dataset_yaml_generates_correct_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            dirs = SplitOutputDirs.for_dataset_root(dataset_root)
            classes = ["head", "reflective", "person"]

            yaml_path = write_dataset_yaml(
                dataset_root=dataset_root,
                classes=classes,
                dirs=dirs,
            )

            self.assertTrue(yaml_path.exists())
            content = yaml_path.read_text(encoding="utf-8")

            self.assertIn(f"path: {dataset_root}", content)
            self.assertIn("train: train/images", content)
            self.assertIn("val: val/images", content)
            self.assertIn("test: test/images", content)
            self.assertIn("nc: 3", content)
            self.assertIn("0: head", content)
            self.assertIn("1: reflective", content)
            self.assertIn("2: person", content)

    def test_write_dataset_yaml_respects_custom_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "ds"
            custom = root / "custom" / "my.yaml"
            dirs = SplitOutputDirs.for_dataset_root(dataset_root)
            classes = ["ship"]

            result = write_dataset_yaml(
                dataset_root=dataset_root,
                classes=classes,
                dirs=dirs,
                output_path=custom,
            )

            self.assertEqual(result, custom)
            self.assertTrue(custom.exists())

    def test_write_dataset_yaml_includes_odp_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            yaml_path = root / "configs" / "dataset.yaml"

            result = write_dataset_yaml(
                yaml_path,
                dataset_root=dataset_root,
                classes=["ship"],
                manifest=SplitManifest(strategy=SplitStrategy.RANDOM, random_state=42),
                dataset_name="demo",
                source_format=AnnotationFormat.PASCAL_VOC,
                task=Task.DETECT,
            )

            content = result.read_text(encoding="utf-8")
            self.assertIn("odp_meta:", content)
            self.assertIn("dataset_name: demo", content)
            self.assertIn("source_format: pascal_voc", content)
            self.assertIn("strategy: random", content)
            self.assertIn("random_state: 42", content)

    def test_yaml_rel_paths_returns_expected_keys(self) -> None:
        dirs = SplitOutputDirs.for_dataset_root(Path("/tmp/ds"))

        rel = dirs.yaml_rel_paths()

        self.assertEqual(rel["train"], "train/images")
        self.assertEqual(rel["val"], "val/images")
        self.assertEqual(rel["test"], "test/images")


if __name__ == "__main__":
    unittest.main()
