"""pair_existence check — 验证每张图都有对应的 .txt 标签文件。

Severity 按缺失比例分级:
    missing_ratio == 0.0           → PASS
    missing_ratio <  WARN_RATIO    → INFO    (个别遗漏, 可容忍)
    missing_ratio <  ERROR_RATIO   → WARNING (显著遗漏, 影响精度)
    missing_ratio >= ERROR_RATIO   → ERROR   (流程级问题, 不允许训练)

注: 空 .txt 文件算"存在"且合法 (YOLO 接受空标签表示无目标图像)。
    这里只判定"文件是否存在", 不判定"是否为空"。
"""

from __future__ import annotations

from typing import Any

from od_platform.common.constants import (
    PAIR_MISSING_ERROR_RATIO,
    PAIR_MISSING_WARN_RATIO,
)
from od_platform.data_validation.checks import _skip_if_yaml_invalid
from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)

DETAILS_PREVIEW_LIMIT = 20


@check("pair_existence")
def check_pair_existence(ctx: CheckContext) -> CheckResult:
    """Check every split image has a same-stem txt label."""
    skipped = _skip_if_yaml_invalid(ctx, "pair_existence")
    if skipped is not None:
        return skipped

    snapshot = ctx.snapshot
    assert snapshot is not None

    if not snapshot.images_per_split:
        return CheckResult(
            name="pair_existence",
            severity=CheckSeverity.INFO,
            summary="无任何 split 可检查 (snapshot 为空)",
            details={"reason": "empty_snapshot"},
        )

    orphan_per_split: dict[str, list[str]] = {}
    total_images = 0
    total_missing = 0

    for split, images in snapshot.images_per_split.items():
        labels = snapshot.labels_per_split.get(split, ())
        missing_in_split: list[str] = []
        for img, lbl in zip(images, labels):
            total_images += 1
            if not lbl.exists():
                total_missing += 1
                missing_in_split.append(str(img))
        if missing_in_split:
            orphan_per_split[split] = missing_in_split

    missing_ratio = total_missing / max(total_images, 1)

    if total_missing == 0:
        severity = CheckSeverity.PASS
        summary = f"全部 {total_images} 张图像都有对应标签"
    elif missing_ratio >= PAIR_MISSING_ERROR_RATIO:
        severity = CheckSeverity.ERROR
        summary = (
            f"缺标签比例 {missing_ratio:.1%} ≥ {PAIR_MISSING_ERROR_RATIO:.0%} "
            f"({total_missing}/{total_images} 张图无标签)"
        )
    elif missing_ratio >= PAIR_MISSING_WARN_RATIO:
        severity = CheckSeverity.WARNING
        summary = (
            f"缺标签比例 {missing_ratio:.1%} ≥ {PAIR_MISSING_WARN_RATIO:.0%} "
            f"({total_missing}/{total_images} 张图无标签)"
        )
    else:
        severity = CheckSeverity.INFO
        summary = f"少量标签缺失 ({total_missing}/{total_images} = {missing_ratio:.2%})"

    details: dict[str, Any] = {
        "total_images": total_images,
        "missing_labels": total_missing,
        "missing_ratio": round(missing_ratio, 4),
        "thresholds": {
            "error_at": PAIR_MISSING_ERROR_RATIO,
            "warn_at": PAIR_MISSING_WARN_RATIO,
        },
        "missing_per_split": {
            split: len(orphans) for split, orphans in orphan_per_split.items()
        },
        "action": "补齐缺失 label；背景图也需保留同名空 txt",
    }

    if orphan_per_split:
        details["missing_examples"] = {
            split: orphans[:DETAILS_PREVIEW_LIMIT]
            for split, orphans in orphan_per_split.items()
        }

    return CheckResult(
        name="pair_existence",
        severity=severity,
        summary=summary,
        details=details,
    )
