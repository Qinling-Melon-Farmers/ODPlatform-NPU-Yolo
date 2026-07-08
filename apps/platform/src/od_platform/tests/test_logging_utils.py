import logging
import tempfile
import unittest
from pathlib import Path

from od_platform.common.logging_utils import (
    build_console_formatter,
    build_file_formatter,
    format_log_rule,
    format_log_section,
    format_status_label,
    get_logger,
)


class TestLoggingUtils(unittest.TestCase):
    def _close_test_loggers(self) -> None:
        for name in list(logging.Logger.manager.loggerDict):
            if name.startswith("od_platform.tests.logging_utils"):
                logger = logging.getLogger(name)
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)

    def tearDown(self) -> None:
        self._close_test_loggers()

    def test_log_style_helpers_return_stable_ascii(self) -> None:
        self.assertEqual(format_log_rule(4, "-"), "----")
        self.assertEqual(format_log_section("Title", 11, "="), "== Title ==")
        self.assertEqual(format_status_label("created"), "[OK]")
        self.assertEqual(format_status_label("custom"), "[CUSTOM]")

    def test_file_log_has_no_ansi_escape_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = get_logger(
                base_path=Path(temp_dir),
                log_type="unit",
                logger_name="od_platform.tests.logging_utils.file_plain",
            )

            logger.warning("plain file log message")

            log_file = next((Path(temp_dir) / "unit").glob("*.log"))
            content = log_file.read_text(encoding="utf-8")
            self._close_test_loggers()

        self.assertNotIn("\x1b[", content)
        self.assertIn("plain file log message", content)

    def test_console_formatter_emits_ansi_but_file_formatter_does_not(self) -> None:
        record = logging.LogRecord(
            name="od_platform.tests.logging_utils.record",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="colored console message",
            args=(),
            exc_info=None,
        )

        console_text = build_console_formatter().format(record)
        file_text = build_file_formatter().format(record)

        self.assertIn("\x1b[", console_text)
        self.assertNotIn("\x1b[", file_text)
        self.assertIn("colored console message", console_text)


if __name__ == "__main__":
    unittest.main()
