"""Dataset snapshot builder for validation checks.

The snapshot is the shared, immutable input for checks. It parses the
dataset yaml once, scans split directories once, and stores small side
statistics that are cheap to collect while files are already being visited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from od_platform.common.constants import IMAGE_EXTENSIONS, Task
from od_platform.common.performance_utils import time_it

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitStats:
    """Lightweight per-split statistics collected during the snapshot scan."""

    image_count: int
    annotated_count: int
    total_instances: int
    class_instances: dict[int, int] = field(default_factory=dict)
    class_images: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSnapshot:
    """A reusable dataset view shared by every validation check."""

    yaml_path: Path
    yaml_data: dict[str, Any]
    yaml_load_error: str | None
    data_root: Path
    nc: int | None
    class_names: tuple[str, ...]
    task_type: str
    images_per_split: dict[str, tuple[Path, ...]]
    labels_per_split: dict[str, tuple[Path, ...]]
    label_files_per_split: dict[str, tuple[Path, ...]]
    stats_per_split: dict[str, SplitStats]
    scan_warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(split for split in SPLITS if split in self.images_per_split)

    @property
    def total_images(self) -> int:
        return sum(len(images) for images in self.images_per_split.values())


@time_it(name="构建数据集快照", logger_instance=logger, iterations=1)
def build_snapshot(yaml_path: Path, task_type: str | None = None) -> DatasetSnapshot:
    """Build one best-effort snapshot without raising data-quality errors."""

    yaml_path = yaml_path.resolve()
    warnings: list[str] = []
    yaml_data, yaml_error = _load_yaml(yaml_path)
    if yaml_error is not None:
        warnings.append(yaml_error)

    data_root = _resolve_data_root(yaml_path, yaml_data)
    nc = yaml_data.get("nc") if isinstance(yaml_data.get("nc"), int) else None
    class_names = _normalize_names(yaml_data.get("names"))
    resolved_task = _resolve_task(yaml_data, task_type, warnings)

    images_per_split: dict[str, tuple[Path, ...]] = {}
    labels_per_split: dict[str, tuple[Path, ...]] = {}
    label_files_per_split: dict[str, tuple[Path, ...]] = {}
    stats_per_split: dict[str, SplitStats] = {}

    for split in SPLITS:
        if split not in yaml_data:
            continue

        split_dir = _resolve_split_dir(data_root, yaml_data.get(split))
        if split_dir is None:
            warnings.append(f"{split} split path is empty or invalid")
            continue
        if not split_dir.is_dir():
            warnings.append(f"{split} image directory does not exist: {split_dir}")
            continue

        images = tuple(_list_images(split_dir))
        if not images:
            warnings.append(f"{split} image directory has no images: {split_dir}")

        expected_labels = tuple(_label_path_for_image(image_path) for image_path in images)
        labels_dir = _infer_labels_dir(split_dir)
        label_files = tuple(_list_label_files(labels_dir))

        images_per_split[split] = images
        labels_per_split[split] = expected_labels
        label_files_per_split[split] = label_files
        stats_per_split[split] = _build_split_stats(expected_labels, image_count=len(images))

    snapshot = DatasetSnapshot(
        yaml_path=yaml_path,
        yaml_data=yaml_data,
        yaml_load_error=yaml_error,
        data_root=data_root,
        nc=nc,
        class_names=class_names,
        task_type=resolved_task,
        images_per_split=images_per_split,
        labels_per_split=labels_per_split,
        label_files_per_split=label_files_per_split,
        stats_per_split=stats_per_split,
        scan_warnings=tuple(warnings),
    )
    logger.info("数据集快照完成: yaml=%s, splits=%s, images=%d", yaml_path, snapshot.splits, snapshot.total_images)
    return snapshot


def _load_yaml(yaml_path: Path) -> tuple[dict[str, Any], str | None]:
    if not yaml_path.exists():
        return {}, f"yaml file does not exist: {yaml_path}"

    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return {}, f"yaml parse failed: {exc}"
    except OSError as exc:
        return {}, f"yaml read failed: {exc}"

    if not isinstance(data, dict):
        return {}, f"yaml top-level must be a dict, got {type(data).__name__}"
    return data, None


def _resolve_data_root(yaml_path: Path, yaml_data: dict[str, Any]) -> Path:
    raw = yaml_data.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return yaml_path.parent.resolve()

    root = Path(raw)
    return root.resolve() if root.is_absolute() else (yaml_path.parent / root).resolve()


def _resolve_split_dir(data_root: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None

    path = Path(raw)
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def _resolve_task(yaml_data: dict[str, Any], task_type: str | None, warnings: list[str]) -> str:
    raw = task_type or yaml_data.get("task") or Task.DETECT
    task = str(raw)
    if task not in Task.all():
        warnings.append(f"invalid task type {task!r}, fallback to {Task.DETECT!r}")
        return Task.DETECT
    return task


def _normalize_names(names_raw: Any) -> tuple[str, ...]:
    if isinstance(names_raw, list):
        if all(isinstance(name, str) and name for name in names_raw):
            return tuple(names_raw)
        return ()

    if isinstance(names_raw, dict):
        try:
            pairs = sorted(((int(key), value) for key, value in names_raw.items()), key=lambda item: item[0])
        except (TypeError, ValueError):
            return ()
        if all(isinstance(value, str) and value for _, value in pairs):
            return tuple(value for _, value in pairs)
    return ()


def _list_images(split_dir: Path) -> list[Path]:
    suffixes = {suffix.lower() for suffix in IMAGE_EXTENSIONS}
    return sorted(path for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _infer_labels_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return image_dir.parent / "labels"


def _label_path_for_image(image_path: Path) -> Path:
    labels_dir = _infer_labels_dir(image_path.parent)
    return labels_dir / f"{image_path.stem}.txt"


def _list_label_files(labels_dir: Path) -> list[Path]:
    if not labels_dir.is_dir():
        return []
    return sorted(path for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")


def _build_split_stats(labels: tuple[Path, ...], *, image_count: int) -> SplitStats:
    annotated_count = 0
    total_instances = 0
    class_instances: dict[int, int] = {}
    class_images: dict[int, int] = {}

    for label_path in labels:
        if not label_path.exists():
            continue
        try:
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            continue

        if lines:
            annotated_count += 1

        seen_classes: set[int] = set()
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            total_instances += 1
            class_instances[class_id] = class_instances.get(class_id, 0) + 1
            seen_classes.add(class_id)

        for class_id in seen_classes:
            class_images[class_id] = class_images.get(class_id, 0) + 1

    return SplitStats(
        image_count=image_count,
        annotated_count=annotated_count,
        total_instances=total_instances,
        class_instances=class_instances,
        class_images=class_images,
    )
