"""Compatibility exports for the split strategy registry."""

from __future__ import annotations

from od_platform.data_pipeline.split.registry import (
    SplitFunc,
    SplitOptions,
    SplitterEntry,
    get_splitter,
    list_strategies,
    register,
)

__all__ = [
    "SplitFunc",
    "SplitOptions",
    "SplitterEntry",
    "get_splitter",
    "list_strategies",
    "register",
]
