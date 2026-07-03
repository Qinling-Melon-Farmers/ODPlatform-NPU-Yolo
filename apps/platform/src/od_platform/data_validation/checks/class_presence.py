"""class_presence check — 验证每个声明的类别在每个 split 中至少出现一次。

使用 snapshot 已统计的 class_instances (零额外 I/O)。
Severity: 某类别在某个 split 中完全消失 → ERROR。
"""

from __future__ import annotations

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("class_presence")
def check_class_presence(ctx: CheckContext) -> CheckResult:
    """Check that each declared class appears in each split."""
    snapshot = ctx.snapshot
    if snapshot is None:
        return CheckResult(
            name="class_presence",
            severity=CheckSeverity.INFO,
            summary="snapshot 未初始化，跳过类别存在性检查",
            details={"reason": "snapshot_missing"},
        )

    if not snapshot.class_names:
        return CheckResult(
            name="class_presence",
            severity=CheckSeverity.INFO,
            summary="未声明任何类别，跳过类别存在性检查",
            details={"reason": "no_classes_declared"},
        )

    if not snapshot.stats_per_split:
        return CheckResult(
            name="class_presence",
            severity=CheckSeverity.INFO,
            summary="快照中没有 split 统计，跳过类别存在性检查",
            details={"reason": "no_split_stats"},
        )

    problems: list[str] = []
    class_names = snapshot.class_names

    for split, stat in snapshot.stats_per_split.items():
        for idx, name in enumerate(class_names):
            count = stat.class_instances.get(idx, 0)
            if count == 0:
                problems.append(f"{name} (class_id={idx}) 在 {split} 中实例数为 0")

    if not problems:
        return CheckResult(
            name="class_presence",
            severity=CheckSeverity.PASS,
            summary=f"所有 {len(class_names)} 个类别在各 split 中至少出现 1 次",
            details={"class_count": len(class_names)},
        )

    return CheckResult(
        name="class_presence",
        severity=CheckSeverity.ERROR,
        summary=f"{len(problems)} 个类别在某个 split 中完全缺失",
        details={
            "reason": "class_missing_in_split",
            "problems": problems,
            "action": "检查标注数据是否覆盖所有类别；确认数据划分策略对稀有类公平",
        },
    )
