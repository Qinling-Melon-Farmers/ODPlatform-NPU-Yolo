import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from od_platform.cli.validate_data import main as validate_main
from od_platform.data_validation.registry import (
    CheckContext,
    CheckEntry,
    CheckResult,
    CheckSeverity,
    get_all_checks,
    list_check_names,
)
from od_platform.data_validation.service import run_all_checks, validate_dataset
from od_platform.validate_dateset.registry import CheckContext as CompatCheckContext
from od_platform.validate_dateset.registry import list_check_names as compat_list_check_names
from od_platform.validate_dateset.service import run_all_checks as compat_run_all_checks


class TestDataValidation(unittest.TestCase):
    def _make_dataset(
        self,
        root: Path,
        *,
        bad_label: bool = False,
        missing_label: bool = False,
        duplicate_stem: bool = False,
    ) -> Path:
        dataset_root = root / "dataset"
        yaml_path = root / "configs" / "demo.yaml"
        for split in ("train", "val", "test"):
            image_dir = dataset_root / split / "images"
            label_dir = dataset_root / split / "labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            stem = "shared_001" if duplicate_stem and split in ("train", "val") else f"{split}_001"
            (image_dir / f"{stem}.jpg").write_bytes(b"image")
            if split != "val" or not missing_label:
                line = "3 0.5 0.5 0.1 0.1\n" if bad_label else "0 0.5 0.5 0.1 0.1\n"
                (label_dir / f"{stem}.txt").write_text(line, encoding="utf-8")

        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(
            "\n".join(
                [
                    f"path: {dataset_root.as_posix()}",
                    "train: train/images",
                    "val: val/images",
                    "test: test/images",
                    "nc: 1",
                    "names:",
                    "  0: ship",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return yaml_path

    def test_registry_auto_imports_builtin_checks(self) -> None:
        names = list_check_names()

        self.assertIn("yaml_schema", names)
        self.assertIn("pair_existence", names)
        self.assertIn("label_format", names)
        self.assertIn("split_uniqueness", names)
        self.assertIn("orphan_labels", names)
        self.assertIn("class_presence", names)
        self.assertIn("annotation_coverage", names)

    def test_validate_dataset_passes_clean_yolo_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir))

            report = validate_dataset(yaml_path=yaml_path, write_report=False)

            self.assertEqual(report.exit_code, 0)
            self.assertTrue(all(result.severity in (CheckSeverity.PASS, CheckSeverity.INFO) for result in report.results))

    def test_validate_dataset_reports_error_and_writes_fix_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            yaml_path = self._make_dataset(root, bad_label=True)
            with patch("od_platform.data_validation.service.paths.RUNS_DIR", root / "runs"):
                report = validate_dataset(yaml_path=yaml_path, write_report=True)

            self.assertEqual(report.exit_code, 2)
            self.assertIsNotNone(report.report_path)
            payload = json.loads((report.report_path or Path()).read_text(encoding="utf-8"))
            self.assertEqual(payload["exit_code"], 2)
            self.assertTrue(payload["fix_items"])

    def test_missing_label_is_reported_by_pair_existence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir), missing_label=True)

            report = validate_dataset(yaml_path=yaml_path, write_report=False)

            result = next(item for item in report.results if item.name == "pair_existence")
            self.assertNotEqual(result.severity, CheckSeverity.PASS)
            self.assertEqual(result.details["missing_labels"], 1)

    def test_split_uniqueness_detects_duplicate_stems(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir), duplicate_stem=True)

            report = validate_dataset(yaml_path=yaml_path, write_report=False)

            self.assertEqual(report.exit_code, 2)
            result = next(item for item in report.results if item.name == "split_uniqueness")
            self.assertEqual(result.severity, CheckSeverity.ERROR)

    def test_check_context_builds_snapshot_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir))

            ctx = CheckContext(yaml_path=yaml_path)

            self.assertIs(ctx.snapshot, ctx.snapshot)
            self.assertEqual(ctx.snapshot.total_images if ctx.snapshot is not None else 0, 3)
            self.assertEqual(ctx.classes, ["ship"])

    def test_single_check_exception_does_not_stop_scheduler(self) -> None:
        def broken(_ctx: CheckContext) -> CheckResult:
            raise RuntimeError("boom")

        def ok(_ctx: CheckContext) -> CheckResult:
            return CheckResult("ok", CheckSeverity.PASS, "ok")

        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir))
            ctx = CheckContext(yaml_path=yaml_path)

        with patch(
            "od_platform.data_validation.service.get_all_checks",
            return_value=[CheckEntry("broken", broken), CheckEntry("ok", ok)],
        ):
            results = run_all_checks(ctx)

        self.assertEqual([result.name for result in results], ["broken", "ok"])
        self.assertEqual(results[0].severity, CheckSeverity.ERROR)
        self.assertEqual(results[1].severity, CheckSeverity.PASS)

    def test_validate_dateset_compatibility_import_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir))
            ctx = CompatCheckContext(yaml_path=yaml_path)

            results = compat_run_all_checks(ctx)

        self.assertIn("yaml_schema", compat_list_check_names())
        self.assertTrue(results)

    def test_cli_returns_dataset_error_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir), bad_label=True)

            code = validate_main(["--yaml", str(yaml_path), "--no-report"])

            self.assertEqual(code, 2)

    def test_orphan_labels_detected(self) -> None:
        """Create a label without image → orphan_labels should report WARNING."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            yaml_path = root / "configs" / "demo.yaml"
            for split in ("train", "val"):
                image_dir = dataset_root / split / "images"
                label_dir = dataset_root / split / "labels"
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                (image_dir / f"{split}_001.jpg").write_bytes(b"image")
                (label_dir / f"{split}_001.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            # orphan label: exists but no corresponding image
            (dataset_root / "train" / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_path.write_text(
                f"path: {dataset_root.as_posix()}\ntrain: train/images\nval: val/images\nnc: 1\nnames:\n  0: ship\n",
                encoding="utf-8",
            )

            report = validate_dataset(yaml_path=yaml_path, write_report=False)

            result = next(item for item in report.results if item.name == "orphan_labels")
            self.assertEqual(result.severity, CheckSeverity.WARNING)

    def test_class_presence_all_classes_present(self) -> None:
        """Clean dataset with all classes in all splits → PASS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir))

            report = validate_dataset(yaml_path=yaml_path, write_report=False)

            result = next(item for item in report.results if item.name == "class_presence")
            self.assertEqual(result.severity, CheckSeverity.PASS)

    def test_annotation_coverage_healthy(self) -> None:
        """Fully annotated dataset → PASS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir))

            report = validate_dataset(yaml_path=yaml_path, write_report=False)

            result = next(item for item in report.results if item.name == "annotation_coverage")
            self.assertEqual(result.severity, CheckSeverity.PASS)

    def test_class_presence_missing_class_in_split(self) -> None:
        """Class only in train, not in val → ERROR."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            yaml_path = root / "configs" / "demo.yaml"

            for split in ("train", "val"):
                (dataset_root / split / "images").mkdir(parents=True)
                (dataset_root / split / "labels").mkdir(parents=True)

            # train: two classes present
            (dataset_root / "train" / "images" / "001.jpg").write_bytes(b"img")
            (dataset_root / "train" / "labels" / "001.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (dataset_root / "train" / "images" / "002.jpg").write_bytes(b"img")
            (dataset_root / "train" / "labels" / "002.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            # val: only class 0 present, class 1 missing
            (dataset_root / "val" / "images" / "001.jpg").write_bytes(b"img")
            (dataset_root / "val" / "labels" / "001.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_path.write_text(
                f"path: {dataset_root.as_posix()}\ntrain: train/images\nval: val/images\nnc: 2\nnames:\n  0: cat\n  1: dog\n",
                encoding="utf-8",
            )

            report = validate_dataset(yaml_path=yaml_path, write_report=False)
            result = next(item for item in report.results if item.name == "class_presence")
            self.assertEqual(result.severity, CheckSeverity.ERROR)
            self.assertIn("problems", result.details)
            self.assertTrue(any("dog" in p for p in result.details["problems"]))

    def test_annotation_coverage_low_triggers_warning(self) -> None:
        """50% unannotated → WARNING."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            yaml_path = root / "configs" / "demo.yaml"

            for split in ("train", "val"):
                (dataset_root / split / "images").mkdir(parents=True)
                (dataset_root / split / "labels").mkdir(parents=True)
            # 2 images, only 1 has a label → 50% unannotated, triggers WARNING (>30%)
            (dataset_root / "train" / "images" / "001.jpg").write_bytes(b"img")
            (dataset_root / "train" / "labels" / "001.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (dataset_root / "train" / "images" / "002.jpg").write_bytes(b"img")
            # 002.jpg has NO label (background image)
            (dataset_root / "val" / "images" / "001.jpg").write_bytes(b"img")
            (dataset_root / "val" / "labels" / "001.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

            yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_path.write_text(
                f"path: {dataset_root.as_posix()}\ntrain: train/images\nval: val/images\nnc: 1\nnames:\n  0: ship\n",
                encoding="utf-8",
            )

            report = validate_dataset(yaml_path=yaml_path, write_report=False)
            result = next(item for item in report.results if item.name == "annotation_coverage")
            self.assertEqual(result.severity, CheckSeverity.WARNING)

    def test_all_builtin_checks_are_registered_once(self) -> None:
        entries = get_all_checks()

        self.assertEqual(len(entries), len({entry.name for entry in entries}))


if __name__ == "__main__":
    unittest.main()
