"""Command line entry for YOLO training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from od_platform.common.environment import warn_cli_if_not_expected_environment
from od_platform.runtime_config.loaders import load_train_config, resolve_runtime_config
from od_platform.runtime_config.train import YOLOTrainConfig
from od_platform.training.service import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-train",
        description="Run YOLO training with auditable logs and model archiving.",
    )
    parser.add_argument("--config", "-c", "--yaml", dest="config", default="train", help="Runtime config name or yaml path.")
    parser.add_argument("--executor", "-e", help="Executor name written into the training manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Write plan/logs only, without starting Ultralytics.")
    parser.add_argument("--model", help="Override model path or model name.")
    parser.add_argument("--data", help="Override dataset yaml name or path.")
    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, help="Override image size.")
    parser.add_argument("--device", help="Override training device, such as 0, cpu or 0,1.")
    parser.add_argument("--lr0", type=float, help="Override initial learning rate.")
    parser.add_argument("--optimizer", help="Override optimizer.")
    parser.add_argument("--workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--project", help="Override Ultralytics project directory.")
    parser.add_argument("--name", help="Override Ultralytics run name.")
    parser.add_argument("--no-archive", action="store_true", help="Do not archive best.pt and last.pt.")
    return parser


def _apply_cli_overrides(config: YOLOTrainConfig, args: argparse.Namespace) -> YOLOTrainConfig:
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key
        in {
            "model",
            "data",
            "epochs",
            "batch",
            "imgsz",
            "device",
            "lr0",
            "optimizer",
            "workers",
            "seed",
            "project",
            "name",
        }
        and value is not None
    }
    if args.no_archive:
        overrides["archive_weights"] = False
    return YOLOTrainConfig.model_validate({**config.model_dump(), **overrides})


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    warn_cli_if_not_expected_environment()
    config_path: Path = resolve_runtime_config(args.config)
    try:
        config = _apply_cli_overrides(load_train_config(config_path), args)
        run_training(config, config_source=config_path, executor=args.executor, dry_run=args.dry_run)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
