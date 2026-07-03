"""annotation_coverage check — 检查各 split 中无标注图占比。

Severity: 无标注图占比超过阈值 → WARNING (可能是标注遗漏批次问题)。
"""

from __future__ import annotations

from od_platform.common.constants import (
    NO_ANNOTATION_ERROR_RATIO,
    NO_ANNOTATION_WARN_RATIO,
)
from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)


@check("annotation_coverage")
def check_annotation_coverage(ctx: CheckContext) -> CheckResult:
    """Check that most images in each split have annotations."""
    snapshot = ctx.snapshot
    if snapshot is None:
        return CheckResult(
            name="annotation_coverage",
            severity=CheckSeverity.INFO,
            summary="snapshot 未初始化，跳过标注覆盖率检查",
            details={"reason": "snapshot_missing"},
        )

    if not snapshot.stats_per_split:
        return CheckResult(
            name="annotation_coverage",
            severity=CheckSeverity.INFO,
            summary="快照中没有 split 统计，跳过标注覆盖率检查",
            details={"reason": "no_split_stats"},
        )

    problems: list[str] = []
    max_unannotated_ratio = 0.0
    all_ok = True

    for split, stat in snapshot.stats_per_split.items():
        if stat.image_count == 0:
            continue
        unannotated = stat.image_count - stat.annotated_count
        ratio = unannotated / stat.image_count
        max_unannotated_ratio = max(max_unannotated_ratio, ratio)

        if ratio >= NO_ANNOTATION_ERROR_RATIO:
            all_ok = False
            problems.append(f"{split}: {unannotated}/{stat.image_count} ({ratio:.1%}) 张无标注 — 严重缺失")
        elif ratio >= NO_ANNOTATION_WARN_RATIO:
            all_ok = False
            problems.append(f"{split}: {unannotated}/{stat.image_count} ({ratio:.1%}) 张无标注")

    if all_ok:
        return CheckResult(
            name="annotation_coverage",
            severity=CheckSeverity.PASS,
            summary="各 split 的标注覆盖率正常",
            details={"max_unannotated_ratio": round(max_unannotated_ratio, 4)},
        )

    # 判断整体 severity：如果有任何 split 超过 ERROR 阈值则 ERROR，否则 WARNING
    overall_severity = (
        CheckSeverity.ERROR if max_unannotated_ratio >= NO_ANNOTATION_ERROR_RATIO else CheckSeverity.WARNING
    )

    return CheckResult(
        name="annotation_coverage",
        severity=overall_severity,
        summary=f"标注覆盖率偏低 (最差 split 无标注率 {max_unannotated_ratio:.1%})",
        details={
            "reason": "low_annotation_coverage",
            "problems": problems,
            "thresholds": {
                "warn_at": NO_ANNOTATION_WARN_RATIO,
                "error_at": NO_ANNOTATION_ERROR_RATIO,
            },
            "action": "检查标注文件是否完整；确认无目标的背景图数量是否合理",
        },
    )
