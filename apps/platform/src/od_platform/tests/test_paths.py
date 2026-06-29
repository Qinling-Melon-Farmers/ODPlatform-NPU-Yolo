import unittest
from pathlib import Path

from od_platform.common.paths import (
    APP_DIR,
    DATA_DIR,
    LOGGING_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    ROOT_DIR,
    RUNS_DIR,
    get_dirs_to_initialize,
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


if __name__ == "__main__":
    unittest.main()
