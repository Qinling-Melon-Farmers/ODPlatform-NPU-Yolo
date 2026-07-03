"""Command line entry for YOLO training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from od_platform.runtime_config.loaders import load_train_config, resolve_runtime_config
from od_platform.training.service import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-train",
        description="Run YOLO training with auditable logs and model archiving.",
    )
    parser.add_argument("--config", "-c", default="train", help="Runtime config name or yaml path.")
    parser.add_argument("--executor", "-e", help="Executor name written into the training manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Write plan/logs only, without starting Ultralytics.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path: Path = resolve_runtime_config(args.config)
    try:
        config = load_train_config(config_path)
        run_training(config, config_source=config_path, executor=args.executor, dry_run=args.dry_run)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
