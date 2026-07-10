import re
from pathlib import Path


def get_display_width(text: str) -> int:
    """Return the display width of text in a monospace terminal."""
    return sum(2 if _is_wide_char(char) else 1 for char in text)


def _is_wide_char(char: str) -> bool:
    """Return True when one character is likely rendered as double width."""
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
    """Pad text to a target display width."""
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
    """Format one table row with display-width aware padding."""
    if aligns is None:
        aligns = ["left"] * len(columns)
    assert len(columns) == len(widths) == len(aligns), "columns, widths and aligns must have the same length"

    parts = [
        pad_to_width(str(col), width, align)
        for col, width, align in zip(columns, widths, aligns, strict=True)
    ]
    return " | ".join(parts)


def format_table_separator(widths: list, char: str = "-") -> str:
    """Return a separator line matching table column widths."""
    total = sum(widths) + 3 * (len(widths) - 1)
    return char * total


def model_slug(model_name: str) -> str:
    """Return a stable filesystem-safe slug for a model name or path."""
    stem = Path(str(model_name)).stem or "model"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-_")
    return slug or "model"
