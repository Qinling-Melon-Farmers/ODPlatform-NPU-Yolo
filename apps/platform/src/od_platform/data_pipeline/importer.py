"""Import teaching/raw dataset archives into ODPlatform raw layout."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from od_platform.common import paths
from od_platform.common.constants import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawImportResult:
    dataset_name: str
    raw_root: Path
    images: int
    annotations: int


def import_voc_zip(zip_path: Path, *, dataset_name: str | None = None, overwrite: bool = False) -> RawImportResult:
    """Import a VOC-style zip into data/raw/<dataset>/images and annotations."""
    source = zip_path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"dataset zip does not exist: {source}")

    resolved_name = dataset_name or _dataset_name_from_zip(source)
    raw_root = paths.RAW_DATA_DIR / resolved_name
    images_dir = raw_root / "images"
    annotations_dir = raw_root / "annotations"
    if raw_root.exists() and any(raw_root.iterdir()) and not overwrite:
        raise FileExistsError(f"raw dataset already exists: {raw_root}")
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    image_suffixes = {suffix.lower() for suffix in IMAGE_EXTENSIONS}
    images = 0
    annotations = 0
    used_image_names: set[str] = set()
    used_annotation_names: set[str] = set()

    with ZipFile(source) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = Path(info.filename)
            if _is_unsafe_member(member):
                raise ValueError(f"unsafe zip member path: {info.filename}")
            suffix = member.suffix.lower()
            if suffix in image_suffixes:
                target_name = _unique_name(member, used_image_names)
                _write_member(archive, info.filename, images_dir / target_name)
                images += 1
            elif suffix == ".xml":
                target_name = _unique_name(member, used_annotation_names)
                _write_member(archive, info.filename, annotations_dir / target_name)
                annotations += 1

    logger.info("imported VOC zip %s -> %s, images=%d annotations=%d", source, raw_root, images, annotations)
    return RawImportResult(dataset_name=resolved_name, raw_root=raw_root, images=images, annotations=annotations)


def _dataset_name_from_zip(zip_path: Path) -> str:
    name = zip_path.stem.lower()
    name = name.replace(".v1i.voc", "")
    return "-".join(part for part in name.replace("_", "-").split() if part)


def _is_unsafe_member(member: Path) -> bool:
    return member.is_absolute() or any(part == ".." for part in member.parts)


def _unique_name(member: Path, used: set[str]) -> str:
    name = member.name
    if name not in used:
        used.add(name)
        return name
    prefix = "_".join(part for part in member.parts[:-1] if part)
    candidate = f"{prefix}_{member.name}" if prefix else member.name
    counter = 1
    while candidate in used:
        candidate = f"{prefix}_{counter}_{member.name}" if prefix else f"{counter}_{member.name}"
        counter += 1
    used.add(candidate)
    return candidate


def _write_member(archive: ZipFile, member_name: str, target: Path) -> None:
    with archive.open(member_name) as src, target.open("wb") as dst:
        dst.write(src.read())
