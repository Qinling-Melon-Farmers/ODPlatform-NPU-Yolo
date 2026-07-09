"""Command line entry for raw dataset archive import."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from od_platform.common.environment import warn_cli_if_not_expected_environment
from od_platform.data_pipeline.importer import import_dataset_zip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odp-import-dataset", description="Import raw dataset archives into data/raw.")
    parser.add_argument("zip_path", type=Path, help="Dataset zip file path.")
    parser.add_argument("--name", help="Dataset name under data/raw.")
    parser.add_argument("--format", default="voc", choices=("voc", "yolo"), help="Archive annotation format.")
    parser.add_argument("--overwrite", action="store_true", help="Allow importing into a non-empty raw dataset directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    warn_cli_if_not_expected_environment()
    try:
        import_dataset_zip(args.zip_path, annotation_format=args.format, dataset_name=args.name, overwrite=args.overwrite)
        return 0
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
