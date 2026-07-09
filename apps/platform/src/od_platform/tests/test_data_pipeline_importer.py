import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from od_platform.data_pipeline.importer import import_voc_zip, import_yolo_zip


class TestDataPipelineImporter(unittest.TestCase):
    def test_import_voc_zip_flattens_images_and_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zip_path = root / "steel surface defect.v1i.voc.zip"
            with ZipFile(zip_path, "w") as archive:
                archive.writestr("train/a.jpg", b"image")
                archive.writestr("train/a.xml", "<annotation />")
                archive.writestr("test/b.jpg", b"image")
                archive.writestr("test/b.xml", "<annotation />")

            with patch("od_platform.data_pipeline.importer.paths.RAW_DATA_DIR", root / "data" / "raw"):
                result = import_voc_zip(zip_path)

            self.assertEqual(result.dataset_name, "steel-surface-defect")
            self.assertEqual(result.images, 2)
            self.assertEqual(result.annotations, 2)
            self.assertTrue((result.raw_root / "images" / "a.jpg").exists())
            self.assertTrue((result.raw_root / "annotations" / "b.xml").exists())

    def test_import_voc_zip_rejects_existing_dataset_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zip_path = root / "demo.zip"
            with ZipFile(zip_path, "w") as archive:
                archive.writestr("train/a.jpg", b"image")
                archive.writestr("train/a.xml", "<annotation />")
            raw_dir = root / "data" / "raw" / "demo"
            raw_dir.mkdir(parents=True)
            (raw_dir / "README.md").write_text("exists", encoding="utf-8")

            with patch("od_platform.data_pipeline.importer.paths.RAW_DATA_DIR", root / "data" / "raw"):
                with self.assertRaises(FileExistsError):
                    import_voc_zip(zip_path, dataset_name="demo")

    def test_import_yolo_zip_flattens_images_labels_and_copies_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            zip_path = root / "liftrace.zip"
            with ZipFile(zip_path, "w") as archive:
                archive.writestr("liftrace/data.yaml", "nc: 2\nnames:\n  0: bridge\n  1: tank\n")
                archive.writestr("liftrace/images/train/a.jpg", b"image")
                archive.writestr("liftrace/labels/train/a.txt", "0 0.5 0.5 0.1 0.2\n")
                archive.writestr("liftrace/images/val/b.jpg", b"image")
                archive.writestr("liftrace/labels/val/b.txt", "1 0.5 0.5 0.2 0.3\n")
                archive.writestr("liftrace/images/train/a.jpg\uf03aZone.Identifier", b"ignored")

            with patch("od_platform.data_pipeline.importer.paths.RAW_DATA_DIR", root / "data" / "raw"):
                result = import_yolo_zip(zip_path)

            self.assertEqual(result.dataset_name, "liftrace")
            self.assertEqual(result.images, 2)
            self.assertEqual(result.annotations, 2)
            self.assertTrue((result.raw_root / "data.yaml").exists())
            self.assertTrue((result.raw_root / "images" / "train_a.jpg").exists())
            self.assertTrue((result.raw_root / "annotations" / "train_a.txt").exists())
            self.assertFalse(any(path.name.endswith("Identifier") for path in (result.raw_root / "images").iterdir()))


if __name__ == "__main__":
    unittest.main()
