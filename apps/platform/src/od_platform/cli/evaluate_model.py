"""Command line entry for YOLO model evaluation."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from od_platform.common import paths
from od_platform.common.environment import warn_cli_if_not_expected_environment
from od_platform.common.logging_utils import get_logger
from od_platform.evaluation import evaluate_yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odp-val",
        description="Evaluate a trained YOLO model. This is separate from odp-validate data checks.",
    )
    parser.add_argument("--config", "-c", "--yaml", dest="config", default="val", help="Runtime config name or yaml path.")
    parser.add_argument("--model", help="Trained model path, archived directory, or checkpoint name.")
    parser.add_argument("--data", help="Dataset yaml name or path.")
    parser.add_argument("--executor", "-e", help="Executor name written into the evaluation audit.")
    parser.add_argument("--split", choices=("train", "val", "test"), help="Dataset split to evaluate.")
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, help="Override image size.")
    parser.add_argument("--workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--device", help="Override validation device, such as 0 or cpu.")
    parser.add_argument("--conf", type=float, help="Override confidence threshold.")
    parser.add_argument("--iou", type=float, help="Override IoU threshold.")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Override max detections per image.")
    parser.add_argument("--project", help="Override Ultralytics project directory.")
    parser.add_argument("--name", help="Override Ultralytics run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow Ultralytics to reuse an existing run directory.")
    parser.add_argument("--half", dest="half", action="store_true", default=None, help="Enable half precision.")
    parser.add_argument("--no-half", dest="half", action="store_false", help="Disable half precision.")
    parser.add_argument("--plots", dest="plots", action="store_true", default=None, help="Enable validation plots.")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="Disable validation plots.")
    parser.add_argument("--save-json", dest="save_json", action="store_true", default=None, help="Save JSON predictions.")
    parser.add_argument("--no-save-json", dest="save_json", action="store_false", help="Do not save JSON predictions.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console/file log level.",
    )
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    keys = {
        "split",
        "batch",
        "imgsz",
        "workers",
        "device",
        "conf",
        "iou",
        "max_det",
        "project",
        "name",
        "half",
        "plots",
        "save_json",
    }
    payload = {key: getattr(args, key) for key in keys if getattr(args, key, None) is not None}
    if args.exist_ok:
        payload["exist_ok"] = True
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # torch + matplotlib 同载 libiomp5md.dll 会触发 OMP Error #15（无害修复）
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    get_logger(
        base_path=paths.LOGGING_DIR,
        log_type="evaluation",
        log_level=getattr(logging, args.log_level),
        temp_log=False,
    )
    warn_cli_if_not_expected_environment(logger=logging.getLogger("od_platform"))
    try:
        result = evaluate_yolo(
            config_path=args.config,
            model=args.model,
            data=args.data,
            cli_overrides=_cli_overrides(args),
            executor=args.executor,
        )
        if not result.success:
            parser.exit(1, f"odp-val failed: {result.error}\n")
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
