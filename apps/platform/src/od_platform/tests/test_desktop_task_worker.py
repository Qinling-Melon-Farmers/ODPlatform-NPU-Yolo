import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

DESKTOP_DIR = Path(__file__).resolve().parents[4] / "desktop"
if str(DESKTOP_DIR) not in sys.path:
    sys.path.insert(0, str(DESKTOP_DIR))

from infer_worker import InferWorker  # noqa: E402
from task_worker import CommandWorker  # noqa: E402


class TestDesktopTaskWorker(unittest.TestCase):
    def test_command_worker_builds_pythonpath_and_streams_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            platform_src = root / "apps" / "platform" / "src"
            outputs: list[str] = []
            completed: list[int] = []
            worker = CommandWorker(
                module="od_platform.cli.evaluate_model",
                args=["--help"],
                cwd=root,
                platform_src=platform_src,
            )
            worker.output_ready.connect(outputs.append)
            worker.completed.connect(completed.append)

            class FakeProcess:
                stdout = iter(["line one\n", "line two\n"])

                def wait(self) -> int:
                    return 0

                def poll(self):
                    return 0

            with patch("subprocess.Popen", return_value=FakeProcess()) as popen:
                worker.run()

            command = popen.call_args.args[0]
            env = popen.call_args.kwargs["env"]
            self.assertEqual(command[:3], [sys.executable, "-m", "od_platform.cli.evaluate_model"])
            self.assertEqual(env["PYTHONPATH"].split(";")[0], str(platform_src))
            self.assertIn("line one", outputs)
            self.assertEqual(completed, [0])

    def test_command_worker_cancel_terminates_running_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            worker = CommandWorker(
                module="od_platform.cli.evaluate_model",
                args=[],
                cwd=Path(temp_dir),
                platform_src=Path(temp_dir),
            )

            class FakeProcess:
                terminated = False

                def poll(self):
                    return None

                def terminate(self) -> None:
                    self.terminated = True

            fake_process = FakeProcess()
            worker._process = fake_process

            worker.cancel()

            self.assertTrue(fake_process.terminated)

    def test_infer_worker_passes_video_stride_to_platform_service(self) -> None:
        worker = InferWorker(
            model="best.pt",
            source="video.mp4",
            conf=0.25,
            iou=0.7,
            device="0",
            name="desktop-test",
            vid_stride=4,
            max_frames=10,
        )

        with patch("infer_worker.infer_yolo", return_value=SimpleNamespace(success=True, error=None)) as infer_yolo:
            result = worker._run_inference()

        self.assertTrue(result.success)
        cli_args = infer_yolo.call_args.kwargs["cli_args"]
        self.assertEqual(cli_args["vid_stride"], 4)
        self.assertEqual(cli_args["task"], "detect")


if __name__ == "__main__":
    unittest.main()
