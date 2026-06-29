import unittest

from od_platform.common.string_utils import (
    format_table_row,
    format_table_separator,
    get_display_width,
    pad_to_width,
)


class TestStringUtils(unittest.TestCase):
    def test_display_width_counts_cjk_as_two(self) -> None:
        self.assertEqual(get_display_width("hello"), 5)
        self.assertEqual(get_display_width("你好"), 4)
        self.assertEqual(get_display_width("hi你好"), 6)
        self.assertEqual(get_display_width(""), 0)

    def test_pad_to_width_respects_display_width(self) -> None:
        value = pad_to_width("你好", 10)
        self.assertEqual(get_display_width(value), 10)

    def test_format_table_helpers_use_matching_widths(self) -> None:
        row = format_table_row(["目录", "状态"], [10, 6])
        separator = format_table_separator([10, 6])
        self.assertEqual(get_display_width(row), get_display_width(separator))


if __name__ == "__main__":
    unittest.main()
