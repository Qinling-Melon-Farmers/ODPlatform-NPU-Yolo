"""数据集划分服务层。

@FileName:   service.py
@Function:   样本配对与划分策略调度
"""

from __future__ import annotations

from pathlib import Path

from od_platform.data_pipeline.split.manifest import PairList, SplitManifest
from od_platform.data_pipeline.split.registry import SplitOptions, get_splitter

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_yolo_pairs(
    images_dir: Path,
    labels_dir: Path,
    image_suffixes: tuple[str, ...] = IMAGE_SUFFIXES,
) -> PairList:
    """收集 YOLO 图像与同名 label 配对。

    Args:
        images_dir: 图像目录。
        labels_dir: YOLO label 目录。
        image_suffixes: 支持的图像扩展名。

    Returns:
        ``(image_path, label_path)`` 列表，按文件名稳定排序。

    Raises:
        FileNotFoundError: 目录不存在、没有图片，或图片缺少同名 label 时抛出。
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

    suffixes = {suffix.lower() for suffix in image_suffixes}
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in suffixes)
    if not images:
        raise FileNotFoundError(f"图片目录为空或无支持格式: {images_dir}")

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
    strategy: str,
    options: SplitOptions | None = None,
) -> SplitManifest:
    """按指定策略划分样本对。"""
    entry = get_splitter(strategy)
    return entry.func(pairs, options or SplitOptions())
