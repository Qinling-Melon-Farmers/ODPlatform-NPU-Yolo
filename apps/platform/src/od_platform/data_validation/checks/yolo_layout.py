"""Checks for YOLO train/val/test layout and labels."""

from __future__ import annotations

from pathlib import Path

from od_platform.common.constants import PAIR_MISSING_ERROR_RATIO, PAIR_MISSING_WARN_RATIO, Task
from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)

_SPLITS = ("train", "val", "test")
_PREVIEW_LIMIT = 20


def _skip_if_yaml_invalid(ctx: CheckContext, name: str) -> CheckResult | None:
    if ctx.snapshot is not None and ctx.snapshot.yaml_load_error is None:
        return None
    return CheckResult(
        name=name,
        severity=CheckSeverity.INFO,
        summary="yaml_schema 未通过，跳过依赖数据集路径的检查",
        details={"reason": "skip_due_to_yaml_error"},
    )


@check("pair_existence")
def check_pair_existence(ctx: CheckContext) -> CheckResult:
    """Check every split image has a same-stem txt label."""
    skipped = _skip_if_yaml_invalid(ctx, "pair_existence")
    if skipped is not None:
        return skipped

    snapshot = ctx.snapshot
    assert snapshot is not None

    missing: list[str] = []
    total_images = 0
    for split in snapshot.splits:
        images = snapshot.images_per_split.get(split, ())
        labels = snapshot.labels_per_split.get(split, ())
        total_images += len(images)
        for image_path, label_path in zip(images, labels, strict=True):
            if not label_path.exists():
                missing.append(f"{split}: {image_path.name} -> {label_path.name}")

    if total_images == 0:
        return CheckResult(
            name="pair_existence",
            severity=CheckSeverity.INFO,
            summary="快照中没有可检查的图像",
            details={"reason": "empty_snapshot", "scan_warnings": list(snapshot.scan_warnings)},
        )

    missing_ratio = len(missing) / total_images
    if not missing:
        return CheckResult(
            name="pair_existence",
            severity=CheckSeverity.PASS,
            summary="所有图像都有对应 label 文件",
            details={"total_images": total_images, "missing_labels": 0},
        )

    severity = CheckSeverity.WARNING
    if missing_ratio >= PAIR_MISSING_ERROR_RATIO:
        severity = CheckSeverity.ERROR
    elif missing_ratio < PAIR_MISSING_WARN_RATIO:
        severity = CheckSeverity.INFO

    return CheckResult(
        name="pair_existence",
        severity=severity,
        summary=f"{len(missing)}/{total_images} ({missing_ratio:.1%}) 个图像缺少 label 文件",
        details={
            "reason": "missing_label_files",
            "problems": missing[:_PREVIEW_LIMIT],
            "total_images": total_images,
            "missing_labels": len(missing),
            "missing_ratio": missing_ratio,
            "action": "补齐缺失 label；背景图也应保留同名空 txt",
        },
    )


@check("label_format")
def check_label_format(ctx: CheckContext) -> CheckResult:
    """Validate YOLO txt rows with cached label paths."""
    skipped = _skip_if_yaml_invalid(ctx, "label_format")
    if skipped is not None:
        return skipped

    snapshot = ctx.snapshot
    assert snapshot is not None
    if snapshot.nc is None:
        return CheckResult(
            name="label_format",
            severity=CheckSeverity.INFO,
            summary="nc 无效，跳过类别 ID 越界检查",
            details={"reason": "skip_due_to_invalid_nc"},
        )

    problems: list[str] = []
    error_kinds: dict[str, int] = {}
    total_files = 0
    total_rows = 0

    for split in _SPLITS:
        for label_path in snapshot.labels_per_split.get(split, ()):
            if not label_path.exists():
                continue
            total_files += 1
            rows, split_problems, split_error_kinds = _validate_label_file(label_path, snapshot.nc, snapshot.task_type)
            total_rows += rows
            problems.extend(f"{split}: {problem}" for problem in split_problems)
            for kind, count in split_error_kinds.items():
                error_kinds[kind] = error_kinds.get(kind, 0) + count

    if problems:
        return CheckResult(
            name="label_format",
            severity=CheckSeverity.ERROR,
            summary=f"发现 {len(problems)} 个 YOLO label 格式问题",
            details={
                "reason": "invalid_label_format",
                "problems": problems[:50],
                "error_kinds": error_kinds,
                "label_files": total_files,
                "rows": total_rows,
                "action": "修正 txt 标注行格式、类别 ID 和归一化坐标",
            },
        )

    return CheckResult(
        name="label_format",
        severity=CheckSeverity.PASS,
        summary="YOLO label 格式合法",
        details={"label_files": total_files, "rows": total_rows, "task": snapshot.task_type},
    )


@check("split_uniqueness")
def check_split_uniqueness(ctx: CheckContext) -> CheckResult:
    """Ensure the same image stem does not appear in multiple splits."""
    skipped = _skip_if_yaml_invalid(ctx, "split_uniqueness")
    if skipped is not None:
        return skipped

    snapshot = ctx.snapshot
    assert snapshot is not None
    stems_by_split = {
        split: {path.stem for path in snapshot.images_per_split.get(split, ())}
        for split in _SPLITS
    }

    problems: list[str] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(stems_by_split[left] & stems_by_split[right])
        if overlap:
            problems.append(f"{left} 与 {right} 重复: {', '.join(overlap[:_PREVIEW_LIMIT])}")

    if problems:
        return CheckResult(
            name="split_uniqueness",
            severity=CheckSeverity.ERROR,
            summary=f"split 间存在重复样本: {len(problems)} 组重复",
            details={"reason": "split_leakage", "problems": problems, "action": "重新划分数据集，确保同一图像只属于一个 split"},
        )

    return CheckResult(
        name="split_uniqueness",
        severity=CheckSeverity.PASS,
        summary="train/val/test 之间没有重复图像 stem",
        details={split: len(stems) for split, stems in stems_by_split.items()},
    )


def _validate_label_file(label_path: Path, nc: int, task_type: str) -> tuple[int, list[str], dict[str, int]]:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return 0, [f"{label_path}:{exc}"], {"read_error": 1}

    problems: list[str] = []
    error_kinds: dict[str, int] = {}
    total_rows = 0

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        total_rows += 1
        parts = line.split()
        if task_type == Task.SEGMENT:
            _validate_segment_row(parts, label_path, line_no, nc, problems, error_kinds)
        else:
            _validate_detect_row(parts, label_path, line_no, nc, problems, error_kinds)

    return total_rows, problems, error_kinds


def _validate_detect_row(
    parts: list[str],
    label_path: Path,
    line_no: int,
    nc: int,
    problems: list[str],
    error_kinds: dict[str, int],
) -> None:
    if len(parts) != 5:
        _add_problem(problems, error_kinds, "field_count", f"{label_path}:{line_no} 不是 5 列")
        return

    parsed = _parse_row(parts, label_path, line_no, problems, error_kinds)
    if parsed is None:
        return
    class_id, values = parsed
    _validate_class_id(class_id, nc, label_path, line_no, problems, error_kinds)

    center_x, center_y, width, height = values
    if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
        _add_problem(problems, error_kinds, "center_out_of_range", f"{label_path}:{line_no} center 坐标越界")
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        _add_problem(problems, error_kinds, "size_out_of_range", f"{label_path}:{line_no} width/height 越界")


def _validate_segment_row(
    parts: list[str],
    label_path: Path,
    line_no: int,
    nc: int,
    problems: list[str],
    error_kinds: dict[str, int],
) -> None:
    if len(parts) < 7 or len(parts[1:]) % 2 != 0:
        _add_problem(problems, error_kinds, "field_count", f"{label_path}:{line_no} segment 行至少需要 3 个点")
        return

    parsed = _parse_row(parts, label_path, line_no, problems, error_kinds)
    if parsed is None:
        return
    class_id, values = parsed
    _validate_class_id(class_id, nc, label_path, line_no, problems, error_kinds)
    if any(value < 0.0 or value > 1.0 for value in values):
        _add_problem(problems, error_kinds, "coord_out_of_range", f"{label_path}:{line_no} polygon 坐标越界")


def _parse_row(
    parts: list[str],
    label_path: Path,
    line_no: int,
    problems: list[str],
    error_kinds: dict[str, int],
) -> tuple[int, list[float]] | None:
    try:
        class_id = int(parts[0])
        values = [float(part) for part in parts[1:]]
    except ValueError:
        _add_problem(problems, error_kinds, "numeric_parse", f"{label_path}:{line_no} 存在非数值字段")
        return None
    return class_id, values


def _validate_class_id(
    class_id: int,
    nc: int,
    label_path: Path,
    line_no: int,
    problems: list[str],
    error_kinds: dict[str, int],
) -> None:
    if class_id < 0 or class_id >= nc:
        _add_problem(problems, error_kinds, "class_out_of_range", f"{label_path}:{line_no} class_id={class_id} 越界")


def _add_problem(problems: list[str], error_kinds: dict[str, int], kind: str, message: str) -> None:
    problems.append(message)
    error_kinds[kind] = error_kinds.get(kind, 0) + 1
