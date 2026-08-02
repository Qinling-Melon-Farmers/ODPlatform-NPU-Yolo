import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

DESKTOP_DIR = Path(__file__).resolve().parents[4] / "desktop"
if str(DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(DESKTOP_DIR))

from services.file_utils import (  # noqa: E402
    filter_paths,
    format_mapping,
    read_csv_rows,
    read_json,
    split_extra_args,
)
from services.paths import (  # noqa: E402
    ROOT_DIR,
    default_results_csv,
    is_relative_to,
)


class TestFileUtils(unittest.TestCase):
    def test_read_json_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.json"
            path.write_text('{"a": 1}', encoding="utf-8")
            self.assertEqual(read_json(path), {"a": 1})

    def test_read_json_error_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text("{broken", encoding="utf-8")
            payload = read_json(path)
            self.assertIn("error", payload)
            self.assertIn("path", payload)

    def test_read_csv_rows_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "r.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch", "map50"])
                writer.writeheader()
                writer.writerow({"epoch": "1", "map50": "0.5"})
            rows = read_csv_rows(path)
            self.assertEqual(rows[0]["map50"], "0.5")

    def test_read_csv_rows_gbk_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "r.csv"
            with path.open("w", encoding="gbk", newline="") as handle:
                handle.write("epoch,备注\n1,中文\n")
            rows = read_csv_rows(path)
            self.assertEqual(rows[0]["备注"], "中文")

    def test_format_mapping_preferred_first(self) -> None:
        lines = format_mapping({"b": 2, "a": 1}, preferred=("a",))
        self.assertIn("a", lines[0])
        self.assertIn("b", lines[1])

    def test_format_mapping_jsonifies_nested(self) -> None:
        lines = format_mapping({"nested": {"x": 1}})
        self.assertIn('"x": 1', lines[0])

    def test_filter_paths_case_insensitive(self) -> None:
        paths = [Path("/a/Report.JSON"), Path("/b/other.txt")]
        result = filter_paths(paths, "report")
        self.assertEqual(result, [paths[0]])

    def test_split_extra_args(self) -> None:
        self.assertEqual(split_extra_args("--imgsz 640 --no-archive"), ["--imgsz", "640", "--no-archive"])

    def test_split_extra_args_failure(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            split_extra_args("'unclosed")
        self.assertIn("追加参数解析失败", str(ctx.exception))


class TestDesktopPaths(unittest.TestCase):
    def test_root_dir_points_to_repo(self) -> None:
        self.assertTrue((ROOT_DIR / ".odp-workspace").exists() or (ROOT_DIR / "apps" / "desktop").exists())
        self.assertEqual(ROOT_DIR.name, "ODPlatform")

    def test_is_relative_to(self) -> None:
        self.assertTrue(is_relative_to(ROOT_DIR / "data" / "x", ROOT_DIR))
        self.assertFalse(is_relative_to(Path("C:/elsewhere/x"), ROOT_DIR))

    def test_default_results_csv_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            from services import paths as service_paths

            with patch.object(service_paths, "ROOT_DIR", Path(temp_dir)):
                self.assertEqual(default_results_csv(), Path(temp_dir) / "runs" / "detect" / "train" / "results.csv")


if __name__ == "__main__":
    unittest.main()
