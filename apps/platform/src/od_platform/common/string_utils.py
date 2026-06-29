def get_display_width(text: str) -> int:
    """计算字符串在等宽终端中的实际显示宽度。"""
    return sum(2 if _is_wide_char(char) else 1 for char in text)


def _is_wide_char(char: str) -> bool:
    """判断单字符是否为宽字符。"""
    if "\u4e00" <= char <= "\u9fff":
        return True
    if "\u3000" <= char <= "\u303f":
        return True
    if "\uff00" <= char <= "\uffef":
        return True
    if "\u3400" <= char <= "\u4dbf":
        return True
    if "\u3040" <= char <= "\u309f":
        return True
    if "\u30a0" <= char <= "\u30ff":
        return True
    if "\uac00" <= char <= "\ud7af":
        return True
    return False


def pad_to_width(text: str, width: int, align: str = "left") -> str:
    """将字符串填充到指定显示宽度。"""
    current = get_display_width(text)
    padding = width - current
    if padding <= 0:
        return text

    if align == "right":
        return " " * padding + text
    if align == "center":
        left = padding // 2
        right = padding - left
        return " " * left + text + " " * right
    return text + " " * padding


def format_table_row(columns: list, widths: list, aligns: list | None = None) -> str:
    """格式化表格一行。"""
    if aligns is None:
        aligns = ["left"] * len(columns)
    assert len(columns) == len(widths) == len(aligns), "列数 / 宽度 / 对齐数必须一致"

    parts = [pad_to_width(str(col), width, align) for col, width, align in zip(columns, widths, aligns)]
    return " | ".join(parts)


def format_table_separator(widths: list, char: str = "-") -> str:
    """生成与表格列宽匹配的分隔线。"""
    total = sum(widths) + 3 * (len(widths) - 1)
    return char * total
