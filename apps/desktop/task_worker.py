"""Background CLI task worker for the desktop workbench."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot


class CommandWorker(QObject):
    """Run an ODPlatform CLI module in a background QThread."""

    output_ready = Signal(str)
    completed = Signal(int)
    failed = Signal(str)

    def __init__(self, *, module: str, args: list[str], cwd: Path, platform_src: Path) -> None:
        super().__init__()
        self._module = module
        self._args = args
        self._cwd = cwd
        self._platform_src = platform_src
        self._process: subprocess.Popen[str] | None = None

    @Slot()
    def run(self) -> None:
        """Run the configured command and stream combined output."""
        command = [sys.executable, "-m", self._module, *self._args]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self._platform_src) if not existing_pythonpath else f"{self._platform_src}{os.pathsep}{existing_pythonpath}"
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        self.output_ready.emit(f"> {' '.join(command)}")
        try:
            self._process = subprocess.Popen(
                command,
                cwd=self._cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self.output_ready.emit(line.rstrip())
            exit_code = self._process.wait()
        except Exception as exc:  # noqa: BLE001 - GUI boundary should report errors.
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        finally:
            self._process = None

        self.completed.emit(exit_code)

    @Slot()
    def cancel(self) -> None:
        """Terminate the running subprocess if it is still active."""
        if self._process is not None and self._process.poll() is None:
            self.output_ready.emit("请求终止任务")
            self._process.terminate()
