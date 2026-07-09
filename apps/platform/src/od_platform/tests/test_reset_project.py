import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from od_platform.cli import reset_project as reset_cli


class TestResetProject(unittest.TestCase):
    def setUp(self) -> None:
        self._close_reset_logger()
        reset_cli.logger.propagate = False
        reset_cli.logger.setLevel(logging.INFO)

    def tearDown(self) -> None:
        self._close_reset_logger()

    def _close_reset_logger(self) -> None:
        for handler in list(reset_cli.logger.handlers):
            reset_cli.logger.removeHandler(handler)
            handler.close()

    def test_scan_dir_ignores_top_level_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text("placeholder", encoding="utf-8")
            (root / "file.txt").write_text("runtime", encoding="utf-8")
            nested = root / "exp"
            nested.mkdir()
            (nested / "metric.txt").write_text("1", encoding="utf-8")

            result = reset_cli._scan_dir(root)

            self.assertEqual(result.file_count, 2)
            self.assertEqual(result.total_bytes, 8)

    def test_reset_project_dry_run_keeps_files_and_writes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            target = base / "runs"
            target.mkdir()
            runtime_file = target / "exp.txt"
            runtime_file.write_text("runtime", encoding="utf-8")
            meta_logging = base / "meta_logging"

            with (
                patch.object(reset_cli.paths, "META_LOGGING_DIR", meta_logging),
                patch.object(reset_cli.paths, "get_dirs_to_reset", return_value=[target]),
                patch.object(reset_cli.paths, "is_protected", return_value=False),
            ):
                exit_code = reset_cli.reset_project()
                self._close_reset_logger()

            self.assertEqual(exit_code, 0)
            self.assertTrue(runtime_file.exists())
            audit_files = list((meta_logging / "reset_project").glob("reset-project_audit_*.log"))
            self.assertEqual(len(audit_files), 1)
            self.assertIn("[AUDIT]", audit_files[0].read_text(encoding="utf-8"))

    def test_reset_project_force_cleans_runtime_but_keeps_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            target = base / "runs"
            target.mkdir()
            placeholder = target / "README.md"
            placeholder.write_text("placeholder", encoding="utf-8")
            runtime_file = target / "exp.txt"
            runtime_file.write_text("runtime", encoding="utf-8")
            nested = target / "exp"
            nested.mkdir()
            (nested / "metric.txt").write_text("1", encoding="utf-8")
            meta_logging = base / "meta_logging"

            with (
                patch.object(reset_cli.paths, "META_LOGGING_DIR", meta_logging),
                patch.object(reset_cli.paths, "get_dirs_to_reset", return_value=[target]),
                patch.object(reset_cli.paths, "is_protected", return_value=False),
            ):
                exit_code = reset_cli.reset_project(yes=True, force=True)
                self._close_reset_logger()

            self.assertEqual(exit_code, 0)
            self.assertTrue(target.exists())
            self.assertTrue(placeholder.exists())
            self.assertFalse(runtime_file.exists())
            self.assertFalse(nested.exists())

    def test_protected_target_fails_fast_without_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            target = base / ".git"
            target.mkdir()
            protected_file = target / "config"
            protected_file.write_text("do not delete", encoding="utf-8")
            meta_logging = base / "meta_logging"

            with (
                patch.object(reset_cli.paths, "META_LOGGING_DIR", meta_logging),
                patch.object(reset_cli.paths, "get_dirs_to_reset", return_value=[target]),
                patch.object(reset_cli.paths, "is_protected", return_value=True),
                patch.object(reset_cli, "_execute_deletion") as execute_deletion,
            ):
                exit_code = reset_cli.reset_project(yes=True, force=True)
                self._close_reset_logger()

            self.assertEqual(exit_code, 2)
            self.assertTrue(protected_file.exists())
            execute_deletion.assert_not_called()

    def test_main_parses_dry_run_yes_conflict_as_safe(self) -> None:
        with patch.object(reset_cli, "reset_project", return_value=0) as reset_project:
            exit_code = reset_cli.main(["--dry-run", "--yes"])

        self.assertEqual(exit_code, 0)
        reset_project.assert_called_once_with(yes=True, force=False, dry_run=True, backup=False)

    def test_reset_project_backup_archives_runtime_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            target = base / "runs"
            target.mkdir()
            runtime_file = target / "exp.txt"
            runtime_file.write_text("runtime", encoding="utf-8")
            meta_logging = base / "meta_logging"
            reset_backup = base / "runs" / "reset_backup"

            with (
                patch.object(reset_cli.paths, "META_LOGGING_DIR", meta_logging),
                patch.object(reset_cli.paths, "ROOT_DIR", base),
                patch.object(reset_cli.paths, "RESET_BACKUP_DIR", reset_backup),
                patch.object(reset_cli.paths, "get_dirs_to_reset", return_value=[target]),
                patch.object(reset_cli.paths, "is_protected", return_value=False),
            ):
                exit_code = reset_cli.reset_project(yes=True, force=True, backup=True)
                self._close_reset_logger()

            self.assertEqual(exit_code, 0)
            self.assertFalse(runtime_file.exists())
            archives = list(reset_backup.glob("*/reset-runtime.zip"))
            self.assertEqual(len(archives), 1)


if __name__ == "__main__":
    unittest.main()
