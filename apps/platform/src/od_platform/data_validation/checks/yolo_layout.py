"""Checks for YOLO train/val/test layout and labels."""

from __future__ import annotations

from pathlib import Path

from od_platform.common.constants import IMAGE_EXTENSIONS
from od_platform.data_validation.registry import (
    CheckContext,
    CheckResult,
    CheckSeverity,
    check,
)

_SPLITS = ("train", "val", "test")


def _images_in(directory: Path) -> list[Path]:
    suffixes = {suffix.lower() for suffix in IMAGE_EXTENSIONS}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


@check("split_dirs_exist")
def check_split_dirs_exist(ctx: CheckContext) -> CheckResult:
    """Check that configured split image and label directories exist."""
    missing: list[str] = []
    existing_splits: list[str] = []
    for split in _SPLITS:
        image_dir = ctx.split_image_dir(split)
        label_dir = ctx.split_label_dir(split)
        if image_dir is None:
            if split != "test":
                missing.append(f"{split}: 未配置")
            continue
        if not image_dir.is_dir():
            missing.append(f"{split}: 图片目录不存在 {image_dir}")
            continue
        if label_dir is None or not label_dir.is_dir():
            missing.append(f"{split}: 标签目录不存在 {label_dir}")
            continue
        existing_splits.append(split)

    if missing:
        return CheckResult(
            name="split_dirs_exist",
            severity=CheckSeverity.ERROR,
            summary="存在缺失的 split 目录",
            details={"missing": missing, "action": "检查数据落盘结构或重新运行 odp-transform"},
        )

    return CheckResult(
        name="split_dirs_exist",
        severity=CheckSeverity.PASS,
        summary="train/val/test 目录结构可用",
        details={"splits": existing_splits},
    )


@check("image_label_pairing")
def check_image_label_pairing(ctx: CheckContext) -> CheckResult:
    """Check every image has a same-stem txt label."""
    missing_labels: list[str] = []
    orphan_labels: list[str] = []
    total_images = 0
    total_labels = 0

    for split in _SPLITS:
        image_dir = ctx.split_image_dir(split)
        label_dir = ctx.split_label_dir(split)
        if image_dir is None or label_dir is None or not image_dir.is_dir() or not label_dir.is_dir():
            continue

        images = _images_in(image_dir)
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        total_images += len(images)
        total_labels += len(labels)
        missing_labels.extend(f"{split}/{stem}.txt" for stem in sorted(image_stems - label_stems))
        orphan_labels.extend(f"{split}/{stem}.txt" for stem in sorted(label_stems - image_stems))

    if missing_labels:
        return CheckResult(
            name="image_label_pairing",
            severity=CheckSeverity.ERROR,
            summary=f"{len(missing_labels)} 张图片缺少同名 label",
            details={
                "missing_labels": missing_labels[:20],
                "orphan_labels": orphan_labels[:20],
                "total_images": total_images,
                "total_labels": total_labels,
                "action": "补齐缺失 label，背景图也应有空 txt",
            },
        )

    severity = CheckSeverity.WARNING if orphan_labels else CheckSeverity.PASS
    summary = (
        f"存在 {len(orphan_labels)} 个无对应图片的 label"
        if orphan_labels
        else "图片与标签一一配对"
    )
    return CheckResult(
        name="image_label_pairing",
        severity=severity,
        summary=summary,
        details={
            "orphan_labels": orphan_labels[:20],
            "total_images": total_images,
            "total_labels": total_labels,
        },
    )


@check("yolo_label_format")
def check_yolo_label_format(ctx: CheckContext) -> CheckResult:
    """Check YOLO txt lines and class id range."""
    errors: list[str] = []
    total_files = 0
    total_boxes = 0
    max_class_id = len(ctx.classes) - 1

    for split in _SPLITS:
        label_dir = ctx.split_label_dir(split)
        if label_dir is None or not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            total_files += 1
            for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                total_boxes += 1
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_path}:{line_no} 不是 5 列")
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(part) for part in parts[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_no} 存在非数值字段")
                    continue
                if class_id < 0 or class_id > max_class_id:
                    errors.append(f"{label_path}:{line_no} class_id={class_id} 越界")
                center_x, center_y, width, height = values
                if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
                    errors.append(f"{label_path}:{line_no} center 坐标越界")
                if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    errors.append(f"{label_path}:{line_no} width/height 越界")

    if errors:
        return CheckResult(
            name="yolo_label_format",
            severity=CheckSeverity.ERROR,
            summary=f"发现 {len(errors)} 个 YOLO label 格式问题",
            details={"errors": errors[:50], "action": "修正 txt 标注行格式和类别 ID"},
        )

    return CheckResult(
        name="yolo_label_format",
        severity=CheckSeverity.PASS,
        summary="YOLO label 格式合法",
        details={"label_files": total_files, "boxes": total_boxes},
    )
