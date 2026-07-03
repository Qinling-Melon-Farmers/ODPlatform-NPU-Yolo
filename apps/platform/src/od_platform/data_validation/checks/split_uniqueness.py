"""split_uniqueness check — 验证 train / val / test 之间无图像名重复。

判重方式: 按图像 stem (文件名不含扩展名)。

Severity: 任何重复 → ERROR。数据泄露的破坏性跟数量无关 — 1 张图重复
就足以让评估指标失真, 是质变型错误, 不按比例分级。
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from od_platform.data_validation.checks import _skip_if_yaml_invalid
from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)

OVERLAP_PREVIEW_LIMIT = 20


@check("split_uniqueness")
def check_split_uniqueness(ctx: CheckContext) -> CheckResult:
    """Ensure the same image stem does not appear in multiple splits."""
    skipped = _skip_if_yaml_invalid(ctx, "split_uniqueness")
    if skipped is not None:
        return skipped

    snapshot = ctx.snapshot
    assert snapshot is not None

    if len(snapshot.images_per_split) < 2:
        return CheckResult(
            name="split_uniqueness",
            severity=CheckSeverity.PASS,
            summary=f"少于 2 个 split, 跳过判重 (当前 splits: {list(snapshot.splits)})",
            details={"reason": "fewer_than_2_splits"},
        )

    stems_by_split: dict[str, set[str]] = {
        split: {img.stem for img in images}
        for split, images in snapshot.images_per_split.items()
    }

    overlaps: list[dict[str, Any]] = []
    for s1, s2 in combinations(stems_by_split.keys(), 2):
        common = stems_by_split[s1] & stems_by_split[s2]
        if common:
            stems_sorted = sorted(common)
            overlaps.append({
                "split_a": s1,
                "split_b": s2,
                "count": len(common),
                "preview": stems_sorted[:OVERLAP_PREVIEW_LIMIT],
            })

    if not overlaps:
        return CheckResult(
            name="split_uniqueness",
            severity=CheckSeverity.PASS,
            summary=(
                f"{len(snapshot.splits)} 个 split "
                f"({' / '.join(snapshot.splits)}) 之间无图像名重复"
            ),
            details={"splits": list(snapshot.splits)},
        )

    total_dup = sum(o["count"] for o in overlaps)
    pairs_str = ", ".join(f"{o['split_a']}↔{o['split_b']}({o['count']})" for o in overlaps)
    return CheckResult(
        name="split_uniqueness",
        severity=CheckSeverity.ERROR,
        summary=f"split 间有 {total_dup} 张图像名重复 — 数据泄露! [{pairs_str}]",
        details={
            "reason": "splits_overlap",
            "splits": list(snapshot.splits),
            "total_duplicates": total_dup,
            "overlaps": overlaps,
            "action": "重新划分数据集，确保同一图像只属于一个 split",
        },
    )
