import logging
import unittest
from io import StringIO

from od_platform.common.performance_utils import _format_duration, time_it


class TestPerformanceUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.stream = StringIO()
        self.logger = logging.getLogger(f"{__name__}.{self._testMethodName}")
        self.logger.handlers.clear()
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(self.stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    def tearDown(self) -> None:
        self.logger.handlers.clear()

    def test_format_duration_unit_boundaries(self) -> None:
        self.assertEqual(_format_duration(0.000999), "999.00 μs")
        self.assertEqual(_format_duration(0.001), "1.00 ms")
        self.assertEqual(_format_duration(0.999), "999.00 ms")
        self.assertEqual(_format_duration(1.0), "1.00 s")
        self.assertEqual(_format_duration(59.999), "60.00 s")
        self.assertEqual(_format_duration(60.0), "1 分 0.00 秒")
        self.assertEqual(_format_duration(3599.999), "60 分 0.00 秒")
        self.assertEqual(_format_duration(3600.0), "1 时 0 分 0.00 秒")

    def test_single_iteration_returns_original_value_and_logs(self) -> None:
        @time_it(name="单次函数", logger_instance=self.logger)
        def add(a: int, b: int) -> int:
            return a + b

        self.assertEqual(add(1, 2), 3)
        self.assertIn("单次函数 耗时:", self.stream.getvalue())

    def test_callable_name_is_resolved_at_runtime(self) -> None:
        @time_it(name=lambda item: f"check: {item}", logger_instance=self.logger)
        def run_check(item: str) -> str:
            return item

        self.assertEqual(run_check("yaml_required_fields"), "yaml_required_fields")
        self.assertIn("check: yaml_required_fields", self.stream.getvalue())

    def test_multiple_iterations_returns_last_value_and_logs_average(self) -> None:
        calls = []

        @time_it(iterations=10, name="重复函数", logger_instance=self.logger)
        def count_calls() -> int:
            calls.append(1)
            return len(calls)

        self.assertEqual(count_calls(), 10)
        output = self.stream.getvalue()
        self.assertIn("重复函数 执行 10 次", output)
        self.assertIn("总耗时:", output)
        self.assertIn("平均耗时:", output)

    def test_exception_propagates_without_timing_log(self) -> None:
        @time_it(name="异常函数", logger_instance=self.logger)
        def raises() -> None:
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            raises()
        self.assertEqual(self.stream.getvalue(), "")

    def test_wraps_preserves_function_metadata(self) -> None:
        @time_it(logger_instance=self.logger)
        def documented_func() -> None:
            """文档字符串。"""

        self.assertEqual(documented_func.__name__, "documented_func")
        self.assertIn("文档字符串", documented_func.__doc__ or "")


if __name__ == "__main__":
    unittest.main()
