"""Resolve user-facing resource references into filesystem paths."""

from __future__ import annotations

from pathlib import Path

from od_platform.common import paths


def resolve_ref(ref: str, *, base_dir: Path, default_suffix: str | None = None) -> Path:
    """Resolve a name or path into an absolute path.

    A plain name is resolved below ``base_dir``. A path-like value, such as
    ``foo/bar`` or an absolute path, is resolved as provided.
    """
    path = Path(ref)
    if path.is_absolute() or len(path.parts) > 1:
        return path.resolve()

    name = ref if default_suffix is None or ref.endswith(default_suffix) else ref + default_suffix
    return (base_dir / name).resolve()


def resolve_dataset(ref: str) -> Path:
    """Resolve a dataset name or path into a raw dataset root."""
    return resolve_ref(ref, base_dir=paths.RAW_DATA_DIR)


def resolve_yaml(ref: str) -> Path:
    """Resolve a dataset config name or yaml path."""
    return resolve_ref(ref, base_dir=paths.DATASET_CONFIGS_DIR, default_suffix=".yaml")


def resolve_model(ref: str) -> Path:
    """Resolve a model name or path.

    Plain filenames are looked up below model asset directories first. If the
    file is not present, the original model reference is returned so
    Ultralytics can still resolve official model names such as ``yolo11n.pt``.
    """
    path = Path(ref)
    if path.is_absolute() or len(path.parts) > 1:
        return path.resolve()

    for base_dir in (
        paths.CHECKPOINTS_DIR,
        paths.TRAINED_MODELS_DIR,
        paths.PRETRAINED_MODELS_DIR,
        paths.ROOT_DIR,
    ):
        candidate = base_dir / path.name
        if candidate.exists():
            return candidate.resolve()
    return path


#: 权重文件后缀。
_WEIGHT_SUFFIXES: tuple[str, ...] = (".pt", ".pth")


def _scan_weight_names(base_dir: Path, *, recursive: bool) -> list[str]:
    """扫描目录下的权重文件名，可选递归。"""
    if not base_dir.exists():
        return []
    if recursive:
        candidates = sorted(
            path
            for path in base_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES
        )
    else:
        candidates = [path for path in sorted(base_dir.iterdir()) if path.suffix.lower() in _WEIGHT_SUFFIXES]
    return [candidate.name for candidate in candidates]


def list_local_model_weights() -> list[str]:
    """列出模型资产目录下已存在的权重文件名。

    ``models/checkpoints``、``models/trained``、``models/pretrained``
    递归扫描（训练归档位于 ``trained/<run>/best.pt`` 子目录）；
    工作区根目录仅扫描顶层，避免全仓库遍历。

    Returns:
        模型目录中实际存在的 .pt/.pth 文件名列表（去重，按目录优先级排序）。
    """
    seen: list[str] = []
    for base_dir, recursive in (
        (paths.CHECKPOINTS_DIR, True),
        (paths.TRAINED_MODELS_DIR, True),
        (paths.PRETRAINED_MODELS_DIR, True),
        (paths.ROOT_DIR, False),
    ):
        for name in _scan_weight_names(base_dir, recursive=recursive):
            if name not in seen:
                seen.append(name)
    return seen


def list_available_models() -> list[str]:
    """列出可用的模型引用名：目录条目 + 本地权重。

    Returns:
        模型引用名列表，目录在前（保持定义顺序），本地权重在后。
    """
    from od_platform.model_catalog.catalog import list_models

    names = [info.name for info in list_models()]
    for local_name in list_local_model_weights():
        if local_name not in names:
            names.append(local_name)
    return names
