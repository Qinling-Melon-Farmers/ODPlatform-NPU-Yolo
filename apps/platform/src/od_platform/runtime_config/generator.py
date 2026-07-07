"""Generate editable runtime configuration templates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from od_platform.common import paths


def default_train_config() -> dict[str, Any]:
    """Return a conservative train.yaml template."""
    return {
        "model": "yolo11n.pt",
        "data": "rsod",
        "task": "detect",
        "epochs": 100,
        "patience": 100,
        "batch": 16,
        "imgsz": 640,
        "workers": 8,
        "device": None,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "plots": True,
        "archive_weights": True,
        "copy_archive": True,
    }


def default_infer_config() -> dict[str, Any]:
    """Return a conservative infer.yaml template."""
    return {
        "model": "train3-20250704-165500-yolo11n-best.pt",
        "source": "data/processed/steel-surface-defect/test/images",
        "task": "detect",
        "imgsz": 640,
        "device": 0,
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 300,
        "save": True,
        "save_txt": True,
        "save_conf": True,
        "show": False,
        "show_labels": True,
        "show_conf": True,
        "show_boxes": True,
    }


def write_train_template(path: Path | None = None, *, overwrite: bool = False) -> Path:
    """Write the default training runtime config."""
    target = path or paths.runtime_config_path("train")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"runtime config already exists: {target}")
    target.write_text(
        yaml.safe_dump(default_train_config(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target


def write_infer_template(path: Path | None = None, *, overwrite: bool = False) -> Path:
    """Write the default inference runtime config."""
    target = path or paths.runtime_config_path("infer")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"runtime config already exists: {target}")
    target.write_text(
        yaml.safe_dump(default_infer_config(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odp-gen-config", description="Generate ODPlatform runtime config templates.")
    parser.add_argument("name", nargs="?", default="train", choices=("train", "infer"), help="Template name.")
    parser.add_argument("--output", "-o", type=Path, help="Output yaml path.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.name == "train":
            write_train_template(args.output, overwrite=args.force)
        elif args.name == "infer":
            write_infer_template(args.output, overwrite=args.force)
        return 0
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
