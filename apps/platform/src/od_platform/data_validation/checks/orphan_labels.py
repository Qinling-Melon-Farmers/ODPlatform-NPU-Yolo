"""orphan_labels check — 检测有标签文件无对应图像的孤儿标注。

Severity: 任何孤儿标签 → WARNING (可能是清理图像时忘记删标签)。

注: 本检查不调用 _skip_if_yaml_invalid()，因为孤儿标签判定只依赖
    snapshot 中的 images_per_split 和 label_files_per_split (纯文件系统
    扫描产物)，不依赖 yaml 配置有效性。即使 yaml 损坏，孤儿标签仍然是
    有意义的诊断信息。
"""

from __future__ import annotations

from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)

PREVIEW_LIMIT = 20


@check("orphan_labels")
def check_orphan_labels(ctx: CheckContext) -> CheckResult:
    """Check for label files without a corresponding image.

    Does not call _skip_if_yaml_invalid() by design: this check only uses
    filesystem-level snapshot data (images_per_split, label_files_per_split)
    and does not depend on yaml validity.
    """
    snapshot = ctx.snapshot
    if snapshot is None:
        return CheckResult(
            name="orphan_labels",
            severity=CheckSeverity.INFO,
            summary="snapshot 未初始化，跳过孤儿标签检查",
            details={"reason": "snapshot_missing"},
        )

    orphans: dict[str, list[str]] = {}
    total_labels = 0

    for split in snapshot.splits:
        image_stems = {img.stem for img in snapshot.images_per_split.get(split, ())}
        label_files = snapshot.label_files_per_split.get(split, ())
        total_labels += len(label_files)
        orphan_list = [str(lbl) for lbl in label_files if lbl.stem not in image_stems]
        if orphan_list:
            orphans[split] = orphan_list

    if not orphans:
        return CheckResult(
            name="orphan_labels",
            severity=CheckSeverity.PASS,
            summary=f"无孤儿标签 ({total_labels} 个标签全部有对应图像)",
            details={"total_labels": total_labels},
        )

    total_orphans = sum(len(paths) for paths in orphans.values())
    return CheckResult(
        name="orphan_labels",
        severity=CheckSeverity.WARNING,
        summary=f"发现 {total_orphans} 个孤儿标签 (有标注无图像)",
        details={
            "reason": "orphan_label_files",
            "orphans_per_split": {s: len(paths) for s, paths in orphans.items()},
            "orphan_examples": {s: paths[:PREVIEW_LIMIT] for s, paths in orphans.items()},
            "action": "删除无对应图像的标签文件，确认图像未遗漏",
        },
    )
