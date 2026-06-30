import unittest
from pathlib import Path

from od_platform.common.paths import (
    APP_DIR,
    DATA_DIR,
    LOGGING_DIR,
    META_LOGGING_DIR,
    MODELS_DIR,
    PRETRAINED_MODELS_DIR,
    PROTECTED_DIRS,
    RAW_DATA_DIR,
    ROOT_DIR,
    RUNS_DIR,
    get_dirs_to_initialize,
    get_dirs_to_reset,
    is_protected,
)


class TestPaths(unittest.TestCase):
    def test_root_dir_points_to_workspace(self) -> None:
        self.assertTrue((ROOT_DIR / ".odp-workspace").exists())

    def test_app_dir_points_to_platform_app(self) -> None:
        self.assertEqual(APP_DIR, ROOT_DIR / "apps" / "platform")

    def test_shared_asset_dirs(self) -> None:
        self.assertEqual(DATA_DIR, ROOT_DIR / "data")
        self.assertEqual(MODELS_DIR, ROOT_DIR / "models")
        self.assertEqual(RUNS_DIR, ROOT_DIR / "runs")

    def test_runtime_dirs_include_core_paths(self) -> None:
        dirs = get_dirs_to_initialize()
        self.assertIsInstance(dirs, list)
        self.assertTrue(all(isinstance(path, Path) for path in dirs))
        self.assertIn(RAW_DATA_DIR, dirs)
        self.assertIn(LOGGING_DIR, dirs)

    def test_reset_dirs_are_runtime_allowlist(self) -> None:
        dirs = get_dirs_to_reset()
        self.assertIsInstance(dirs, list)
        self.assertEqual(len(dirs), 6)
        self.assertIn(RUNS_DIR, dirs)
        self.assertIn(LOGGING_DIR, dirs)
        self.assertNotIn(RAW_DATA_DIR, dirs)
        self.assertNotIn(PRETRAINED_MODELS_DIR, dirs)
        self.assertNotIn(META_LOGGING_DIR, dirs)

    def test_protected_dirs_are_tuple_and_cover_sensitive_paths(self) -> None:
        self.assertIsInstance(PROTECTED_DIRS, tuple)
        self.assertIn(ROOT_DIR, PROTECTED_DIRS)
        self.assertIn(ROOT_DIR / ".git", PROTECTED_DIRS)
        self.assertIn(APP_DIR / "src", PROTECTED_DIRS)
        self.assertIn(RAW_DATA_DIR, PROTECTED_DIRS)
        self.assertIn(PRETRAINED_MODELS_DIR, PROTECTED_DIRS)
        self.assertIn(META_LOGGING_DIR, PROTECTED_DIRS)

    def test_is_protected_blocks_sensitive_and_outside_paths(self) -> None:
        self.assertTrue(is_protected(ROOT_DIR / ".git"))
        self.assertTrue(is_protected(ROOT_DIR / ".git" / "objects"))
        self.assertTrue(is_protected(APP_DIR / "src" / "od_platform"))
        self.assertTrue(is_protected(RAW_DATA_DIR / "dataset_a"))
        self.assertTrue(is_protected(Path("C:/tmp/somewhere")))
        self.assertTrue(is_protected(META_LOGGING_DIR))
        self.assertFalse(is_protected(RUNS_DIR))
        self.assertFalse(is_protected(LOGGING_DIR))

    def test_is_protected_rejects_non_path(self) -> None:
        with self.assertRaises(TypeError):
            is_protected("runs")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
