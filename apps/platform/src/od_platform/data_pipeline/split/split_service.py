"""Compatibility exports for the split service layer."""

from __future__ import annotations

from od_platform.data_pipeline.split.service import collect_yolo_pairs, split_pairs

__all__ = ["collect_yolo_pairs", "split_pairs"]
