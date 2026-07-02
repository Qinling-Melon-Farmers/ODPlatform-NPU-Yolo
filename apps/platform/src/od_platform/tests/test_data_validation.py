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


class TestDataValidation(unittest.TestCase):
    def _make_dataset(self, root: Path, *, bad_label: bool = False, missing_label: bool = False) -> Path:
        dataset_root = root / "dataset"
        yaml_path = root / "configs" / "demo.yaml"
        for split in ("train", "val", "test"):
            image_dir = dataset_root / split / "images"
            label_dir = dataset_root / split / "labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / f"{split}_001.jpg").write_bytes(b"image")
            if split != "val" or not missing_label:
                line = "3 0.5 0.5 0.1 0.1\n" if bad_label else "0 0.5 0.5 0.1 0.1\n"
                (label_dir / f"{split}_001.txt").write_text(line, encoding="utf-8")

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

        self.assertIn("yaml_required_fields", names)
        self.assertIn("split_dirs_exist", names)
        self.assertIn("image_label_pairing", names)
        self.assertIn("yolo_label_format", names)

    def test_validate_dataset_passes_clean_yolo_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir))

            report = validate_dataset(yaml_path=yaml_path, write_report=False)

            self.assertEqual(report.exit_code, 0)
            self.assertTrue(all(result.severity == CheckSeverity.PASS for result in report.results))

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

    def test_missing_label_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir), missing_label=True)

            report = validate_dataset(yaml_path=yaml_path, write_report=False)

            self.assertEqual(report.exit_code, 2)
            self.assertTrue(any(result.name == "image_label_pairing" for result in report.results))

    def test_single_check_exception_does_not_stop_scheduler(self) -> None:
        def broken(_ctx: CheckContext) -> CheckResult:
            raise RuntimeError("boom")

        def ok(_ctx: CheckContext) -> CheckResult:
            return CheckResult("ok", CheckSeverity.PASS, "ok")

        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir))
            ctx = validate_dataset(yaml_path=yaml_path, write_report=False).details.get("ctx")
            if ctx is None:
                ctx = CheckContext(
                    yaml_path=yaml_path,
                    config={"path": str(yaml_path.parent), "names": {0: "ship"}, "nc": 1},
                    dataset_root=yaml_path.parent,
                )

        with patch(
            "od_platform.data_validation.service.get_all_checks",
            return_value=[CheckEntry("broken", broken), CheckEntry("ok", ok)],
        ):
            results = run_all_checks(ctx)

        self.assertEqual([result.name for result in results], ["broken", "ok"])
        self.assertEqual(results[0].severity, CheckSeverity.ERROR)
        self.assertEqual(results[1].severity, CheckSeverity.PASS)

    def test_cli_returns_dataset_error_code(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = self._make_dataset(Path(temp_dir), bad_label=True)

            code = validate_main(["--yaml", str(yaml_path), "--no-report"])

            self.assertEqual(code, 2)

    def test_all_builtin_checks_are_registered_once(self) -> None:
        entries = get_all_checks()

        self.assertEqual(len(entries), len({entry.name for entry in entries}))


if __name__ == "__main__":
    unittest.main()
