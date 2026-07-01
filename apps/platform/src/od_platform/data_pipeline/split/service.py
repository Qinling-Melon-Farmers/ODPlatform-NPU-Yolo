"""Dataset split service layer."""

from __future__ import annotations

from pathlib import Path

from od_platform.common.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLIT_STRATEGY,
    IMAGE_EXTENSIONS,
)
from od_platform.data_pipeline.split.manifest import PairList, SplitManifest
from od_platform.data_pipeline.split.registry import SplitOptions, get_splitter


def collect_yolo_pairs(
    images_dir: Path,
    labels_dir: Path,
    image_suffixes: tuple[str, ...] = IMAGE_EXTENSIONS,
) -> PairList:
    """Collect matching ``(image_path, label_path)`` YOLO pairs."""
    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

    suffixes = {suffix.lower() for suffix in image_suffixes}
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in suffixes)
    if not images:
        raise FileNotFoundError(f"图片目录为空或没有支持的图片格式: {images_dir}")

    pairs: PairList = []
    missing: list[Path] = []
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing.append(label_path)

    if missing:
        preview = ", ".join(str(path.name) for path in missing[:5])
        raise FileNotFoundError(f"存在图片缺少同名 label: {preview}")
    return pairs


def split_pairs(
    pairs: PairList,
    strategy: str = DEFAULT_SPLIT_STRATEGY,
    options: SplitOptions | None = None,
    *,
    train_rate: float = 0.8,
    val_rate: float = 0.1,
    random_state: int = DEFAULT_RANDOM_STATE,
    labels_per_image: dict[str, list[str]] | None = None,
    group_per_image: dict[str, str] | None = None,
) -> SplitManifest:
    """Split paired samples with a registered strategy.

    The positional ``strategy, options`` form is kept for existing callers.
    New CLI/orchestrator code can pass the small keyword set directly.
    """
    entry = get_splitter(strategy)
    resolved_options = options or SplitOptions(
        train_rate=train_rate,
        val_rate=val_rate,
        random_state=random_state,
        labels_per_image=labels_per_image,
        group_per_image=group_per_image,
    )
    if labels_per_image is not None:
        resolved_options.labels_per_image = labels_per_image
    if group_per_image is not None:
        resolved_options.group_per_image = group_per_image

    if entry.requires_labels and not resolved_options.labels_per_image:
        raise ValueError(f"划分策略 {strategy!r} 需要 labels_per_image，但未提供")
    return entry.func(pairs, resolved_options)
