"""Inference service with logs, run artifacts, and audit manifest."""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from od_platform.common import paths
from od_platform.common.logging_utils import build_console_formatter, build_file_formatter
from od_platform.common.refs import resolve_model
from od_platform.common.system_utils import get_basic_device_info
from od_platform.runtime_config.infer import YOLOInferConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceRunPlan:
    """Filesystem layout for one inference run."""

    sequence: int
    timestamp: str
    task: str
    model_slug: str
    source_run_name: str
    audit_run_name: str
    source_run_dir: Path
    ultralytics_project: Path
    log_file: Path


@dataclass(frozen=True)
class InferenceRunResult:
    """Result metadata returned by run_inference."""

    plan: InferenceRunPlan
    dry_run: bool
    manifest_path: Path
    summary_path: Path | None


def build_inference_run_plan(config: YOLOInferConfig, *, now: datetime | None = None) -> InferenceRunPlan:
    """Build output directory and log names for one inference run."""
    sequence = _next_inference_sequence(config.task)
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    model_slug = _model_slug(config.model)
    source_run_name = config.name or f"predict-{sequence}"
    audit_run_name = f"{source_run_name}-{timestamp}-{model_slug}"
    ultralytics_project = Path(config.project).resolve() if config.project else paths.INFERENCE_RUNS_DIR / config.task
    return InferenceRunPlan(
        sequence=sequence,
        timestamp=timestamp,
        task=config.task,
        model_slug=model_slug,
        source_run_name=source_run_name,
        audit_run_name=audit_run_name,
        source_run_dir=ultralytics_project / source_run_name,
        ultralytics_project=ultralytics_project,
        log_file=paths.LOGGING_DIR / "inference" / f"{audit_run_name}.log",
    )


def run_inference(
    config: YOLOInferConfig,
    *,
    config_source: Path | None = None,
    executor: str | None = None,
    dry_run: bool = False,
) -> InferenceRunResult:
    """Run or dry-run YOLO inference."""
    if config.source is None:
        raise ValueError("inference source is required")

    plan = build_inference_run_plan(config)
    run_logger = _configure_inference_logger(plan.log_file)
    model_ref = resolve_model(config.model)
    source_ref = resolve_source(config.source)

    try:
        run_logger.info("inference run: %s", plan.audit_run_name)
        run_logger.info("executor: %s", executor or "unknown")
        run_logger.info("config source: %s", config_source or "direct object")
        run_logger.info("model ref: %s", model_ref)
        run_logger.info("source ref: %s", source_ref)
        run_logger.info("ultralytics output: %s", plan.source_run_dir)
        _log_effective_config(config, run_logger, config_source=config_source)

        if dry_run:
            manifest_path = write_inference_manifest(
                plan,
                config,
                config_source=config_source,
                executor=executor,
                source_ref=source_ref,
                model_ref=model_ref,
                dry_run=True,
            )
            return InferenceRunResult(plan=plan, dry_run=True, manifest_path=manifest_path, summary_path=None)

        kwargs = config.to_ultralytics_kwargs()
        kwargs.update(
            {
                "source": str(source_ref),
                "project": str(plan.ultralytics_project),
                "name": plan.source_run_name,
            }
        )
        run_logger.info("start ultralytics inference")
        start = perf_counter()
        from ultralytics import YOLO

        model = YOLO(str(model_ref))
        results = list(model.predict(**kwargs))
        elapsed_seconds = perf_counter() - start
        summary = summarize_predictions(results, elapsed_seconds=elapsed_seconds)
        summary_path = write_prediction_summary(plan.source_run_dir, summary)
        manifest_path = write_inference_manifest(
            plan,
            config,
            config_source=config_source,
            executor=executor,
            source_ref=source_ref,
            model_ref=model_ref,
            dry_run=False,
            prediction_summary=summary,
        )
        run_logger.info("ultralytics inference finished: images=%s detections=%s", summary["images"], summary["detections"])
        run_logger.info("prediction summary: %s", summary_path)
        return InferenceRunResult(plan=plan, dry_run=False, manifest_path=manifest_path, summary_path=summary_path)
    finally:
        _close_logger(run_logger)


def resolve_source(source: str | Path | int) -> str | Path:
    """Resolve local path sources relative to the workspace root while preserving cameras and URLs."""
    source_text = str(source)
    if source_text.isdigit():
        return source_text
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", source_text):
        return source_text

    source_path = Path(source_text)
    if source_path.is_absolute():
        return source_path.resolve()
    workspace_candidate = paths.ROOT_DIR / source_path
    if workspace_candidate.exists():
        return workspace_candidate.resolve()
    return source_path.resolve()


def summarize_predictions(results: list[Any], *, elapsed_seconds: float) -> dict[str, Any]:
    """Extract a compact, JSON-safe prediction summary."""
    image_summaries: list[dict[str, Any]] = []
    total_detections = 0
    for result in results:
        boxes = getattr(result, "boxes", None)
        count = len(boxes) if boxes is not None else 0
        total_detections += count
        image_summaries.append(
            {
                "path": str(getattr(result, "path", "")),
                "detections": count,
                "speed": _json_safe(getattr(result, "speed", None)),
            }
        )
    return {
        "images": len(results),
        "detections": total_detections,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "items": image_summaries,
    }


def write_prediction_summary(run_dir: Path, summary: dict[str, Any]) -> Path:
    """Write summary.json into the inference run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "prediction_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def write_inference_manifest(
    plan: InferenceRunPlan,
    config: YOLOInferConfig,
    *,
    config_source: Path | None,
    executor: str | None,
    source_ref: str | Path,
    model_ref: Path,
    dry_run: bool,
    prediction_summary: dict[str, Any] | None = None,
) -> Path:
    """Write an audit manifest for one inference run."""
    plan.source_run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": plan.source_run_name,
        "audit_run_name": plan.audit_run_name,
        "dry_run": dry_run,
        "executor": executor,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_source": str(config_source) if config_source else None,
        "config": config.audit_snapshot(),
        "model_ref": str(model_ref),
        "source_ref": str(source_ref),
        "device_info": get_basic_device_info(),
        "outputs": {
            "ultralytics_run_dir": str(plan.source_run_dir),
            "log_file": str(plan.log_file),
        },
        "prediction_summary": prediction_summary,
    }
    output = plan.source_run_dir / "inference_manifest.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def copy_model_to_checkpoints(model_path: Path) -> Path:
    """Copy a model file into models/checkpoints for stable local references."""
    paths.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    target = paths.CHECKPOINTS_DIR / model_path.name
    if model_path.resolve() != target.resolve():
        shutil.copy2(model_path, target)
    return target


def _next_inference_sequence(task: str) -> int:
    pattern = re.compile(r"^predict-(\d+)(?:\D.*)?$")
    base = paths.INFERENCE_RUNS_DIR / task
    if not base.exists():
        return 1
    numbers = []
    for item in base.iterdir():
        match = pattern.match(item.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _model_slug(model_name: str) -> str:
    stem = Path(str(model_name)).stem or "model"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-_")
    return slug or "model"


def _configure_inference_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    run_logger = logging.getLogger(f"od_platform.inference.{log_file.stem}")
    run_logger.handlers.clear()
    run_logger.setLevel(logging.INFO)
    run_logger.propagate = False
    file_formatter = build_file_formatter("%(asctime)s - %(levelname)-8s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    run_logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(build_console_formatter())
    run_logger.addHandler(stream_handler)
    return run_logger


def _log_effective_config(config: YOLOInferConfig, target_logger: logging.Logger, *, config_source: Path | None) -> None:
    target_logger.info("=" * 60)
    target_logger.info("effective inference config")
    source = str(config_source) if config_source else "default/direct"
    for field_name in config.__class__.model_fields:
        target_logger.info("%-20s: %s (source: %s)", field_name, getattr(config, field_name, None), source)
    target_logger.info("=" * 60)


def _close_logger(run_logger: logging.Logger) -> None:
    for handler in list(run_logger.handlers):
        handler.close()
        run_logger.removeHandler(handler)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
