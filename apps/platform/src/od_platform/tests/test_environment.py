import io
import unittest
from unittest.mock import patch

from od_platform.common.environment import (
    is_expected_environment,
    warn_cli_if_not_expected_environment,
)


class TestEnvironmentCheck(unittest.TestCase):
    def test_expected_environment_detects_conda_env_name(self) -> None:
        with patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "odplat"}, clear=True):
            self.assertTrue(is_expected_environment())

    def test_expected_environment_detects_executable_path(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.executable", r"D:\Anaconda3\envs\odplat\python.exe"),
        ):
            self.assertTrue(is_expected_environment())

    def test_cli_warning_is_non_blocking(self) -> None:
        stream = io.StringIO()
        with (
            patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "base"}, clear=True),
            patch("sys.executable", r"D:\Anaconda3\envs\base\python.exe"),
        ):
            result = warn_cli_if_not_expected_environment(stream=stream)

        self.assertFalse(result)
        self.assertIn("conda activate odplat", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
