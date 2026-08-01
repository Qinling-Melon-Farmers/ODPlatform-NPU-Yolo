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


def list_local_model_weights() -> list[str]:
    """列出模型资产目录下已存在的权重文件名。

    Returns:
        模型目录中实际存在的 .pt/.pth 文件名列表（去重，按目录优先级排序）。
    """
    seen: list[str] = []
    for base_dir in (
        paths.CHECKPOINTS_DIR,
        paths.TRAINED_MODELS_DIR,
        paths.PRETRAINED_MODELS_DIR,
        paths.ROOT_DIR,
    ):
        if not base_dir.exists():
            continue
        for candidate in sorted(base_dir.iterdir()):
            if candidate.suffix.lower() in (".pt", ".pth") and candidate.name not in seen:
                seen.append(candidate.name)
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
