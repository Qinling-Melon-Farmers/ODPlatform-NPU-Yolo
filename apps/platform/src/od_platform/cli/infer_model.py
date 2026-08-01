"""Command line entry for YOLO inference."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from od_platform.common import paths
from od_platform.common.constants import Task
from od_platform.common.environment import warn_cli_if_not_expected_environment
from od_platform.common.logging_utils import get_logger
from od_platform.inference.service import infer_yolo, run_inference
from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.runtime_config.loaders import load_infer_config, resolve_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odp-infer", description="Run YOLO inference with logs and audit manifest.")
    parser.add_argument("--config", "-c", "--yaml", dest="config", help="Runtime config name or yaml path.")
    parser.add_argument("--executor", "-e", help="Executor name written into the inference manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Write plan/logs only, without starting Ultralytics.")
    parser.add_argument("--pipeline-yaml", help="Frame-source and visualization config yaml.")
    parser.add_argument(
        "--model",
        help="Override model path or model name. Run odp-list-models to see available models.",
    )
    parser.add_argument("--source", help="Override inference source: image, directory, video, URL or camera index.")
    parser.add_argument("--task", choices=Task.end_to_end(), help="Override task. Segment is reserved for future work.")
    parser.add_argument("--imgsz", type=int, help="Override image size.")
    parser.add_argument("--device", help="Override inference device, such as 0 or cpu.")
    parser.add_argument("--conf", type=float, help="Override confidence threshold.")
    parser.add_argument("--iou", type=float, help="Override NMS IoU threshold.")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Override max detections per image.")
    parser.add_argument("--vid-stride", dest="vid_stride", type=int, help="Process every Nth video frame.")
    parser.add_argument("--classes", help="Comma-separated class ids, such as 0,2,5.")
    parser.add_argument("--project", help="Override Ultralytics project directory.")
    parser.add_argument("--name", help="Override Ultralytics run name.")
    parser.add_argument("--save-txt", action="store_true", help="Save prediction labels as txt.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence in txt labels.")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped detections.")
    parser.add_argument("--show", action="store_true", help="Display prediction windows.")
    parser.add_argument("--no-save", action="store_true", help="Do not save rendered prediction images.")
    parser.add_argument("--no-viz", action="store_true", help="Disable beautified rendering and use native YOLO plot().")
    parser.add_argument("--no-hud", action="store_true", help="Disable FPS and status HUD on display windows.")
    parser.add_argument("--warmup", type=int, default=0, help="Skip first N frames after opening the source.")
    parser.add_argument(
        "--threaded",
        action="store_true",
        help=(
            "Enable the staged inference pipeline. Camera uses a latest-frame read buffer; "
            "file sources use bounded queues."
        ),
    )
    parser.add_argument("--max-frames", type=int, help="Stop after N processed frames; useful for smoke tests.")
    parser.add_argument("--window-name", default="odp-infer", help="OpenCV window title when --show is enabled.")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console/file log level for pipeline inference.",
    )
    return parser


def _load_base_config(config_ref: str | None) -> tuple[YOLOInferConfig, Path | None]:
    if config_ref is None:
        return YOLOInferConfig(), None
    config_path = resolve_runtime_config(config_ref)
    return load_infer_config(config_path), config_path


def _apply_cli_overrides(config: YOLOInferConfig, args: argparse.Namespace) -> YOLOInferConfig:
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key
        in {
            "model",
            "source",
            "task",
            "imgsz",
            "device",
            "conf",
            "iou",
            "max_det",
            "vid_stride",
            "project",
            "name",
        }
        and value is not None
    }
    if args.classes:
        overrides["classes"] = [int(item.strip()) for item in args.classes.split(",") if item.strip()]
    for flag in ("save_txt", "save_conf", "save_crop", "show"):
        if getattr(args, flag):
            overrides[flag] = True
    if args.no_save:
        overrides["save"] = False
    return YOLOInferConfig.model_validate({**config.model_dump(), **overrides})


def _pipeline_cli_args(args: argparse.Namespace) -> dict[str, object]:
    keys = {
        "model",
        "source",
        "task",
        "imgsz",
        "device",
        "conf",
        "iou",
        "max_det",
        "vid_stride",
        "project",
        "name",
    }
    payload = {key: getattr(args, key) for key in keys if getattr(args, key, None) is not None}
    if args.classes:
        payload["classes"] = [int(item.strip()) for item in args.classes.split(",") if item.strip()]
    for flag in ("save_txt", "save_conf", "save_crop", "show"):
        if getattr(args, flag):
            payload[flag] = True
    if args.no_save:
        payload["save"] = False
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    warn_cli_if_not_expected_environment()
    try:
        if args.dry_run:
            base_config, config_path = _load_base_config(args.config)
            config = _apply_cli_overrides(base_config, args)
            run_inference(config, config_source=config_path, executor=args.executor, dry_run=True)
            return 0

        get_logger(
            base_path=paths.LOGGING_DIR,
            log_type="infer",
            log_level=getattr(logging, args.log_level),
            temp_log=False,
        )
        result = infer_yolo(
            yaml_path=args.config,
            pipeline_yaml=args.pipeline_yaml,
            cli_args=_pipeline_cli_args(args),
            beautify=not args.no_viz,
            warmup_frames=args.warmup,
            threaded=args.threaded,
            window_name=args.window_name,
            show_info=not args.no_hud,
            max_frames=args.max_frames,
        )
        if not result.success:
            parser.error(result.error or "inference failed")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
