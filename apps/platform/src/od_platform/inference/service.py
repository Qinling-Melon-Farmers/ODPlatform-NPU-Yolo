"""Inference service with logs, run artifacts, and audit manifest."""

from __future__ import annotations

import json
import logging
import re
import shutil
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from od_platform.common import paths
from od_platform.common.logging_utils import configure_run_logger
from od_platform.common.refs import resolve_model
from od_platform.common.string_utils import model_slug
from od_platform.common.system_utils import get_basic_device_info
from od_platform.inference.cancel import CancelToken, PauseToken
from od_platform.inference.hooks import InferHooks
from od_platform.inference.pipeline import InferStats, SequentialInferencePipeline
from od_platform.inference.pipeline_config import PipelineConfig, load_pipeline_config
from od_platform.inference.sinks import LocalFileSink, NullSink, OutputSink
from od_platform.runtime_config.api import build_infer_config
from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.visualization import (
    BeautifyVisualizer,
    Detection,
    DrawStyle,
    detections_from_yolo_result,
)

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


@dataclass(frozen=True)
class InferResult:
    """Result returned by the D8 frame-by-frame inference service."""

    success: bool
    output_dir: Path
    stats: dict[str, Any]
    infer_time: float | None = None
    saved: bool = False
    error: str | None = None
    audit_path: Path | None = None
    log_path: Path | None = None


def build_inference_run_plan(config: YOLOInferConfig, *, now: datetime | None = None) -> InferenceRunPlan:
    """Build output directory and log names for one inference run."""
    sequence = _next_inference_sequence(config.task)
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    model_slug_value = model_slug(config.model)
    source_run_name = config.name or f"predict-{sequence}"
    audit_run_name = f"{source_run_name}-{timestamp}-{model_slug_value}"
    ultralytics_project = Path(config.project).resolve() if config.project else paths.INFERENCE_RUNS_DIR / config.task
    return InferenceRunPlan(
        sequence=sequence,
        timestamp=timestamp,
        task=config.task,
        model_slug=model_slug_value,
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
    run_logger = configure_run_logger(plan.log_file, logger_prefix="od_platform.inference")
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


class InferService:
    """Orchestrate D5 config, frame sources, YOLO inference, visualization and sinks."""

    def predict(
        self,
        yaml_path: str | Path | None = None,
        pipeline_yaml: str | Path | None = None,
        cli_args: dict[str, Any] | Namespace | None = None,
        *,
        beautify: bool = True,
        threaded: bool = False,
        warmup_frames: int = 0,
        window_name: str = "odp-infer",
        show_info: bool = True,
        output_sink: OutputSink | None = None,
        hooks: InferHooks | None = None,
        cancel_token: CancelToken | None = None,
        pause_token: PauseToken | None = None,
        max_frames: int | None = None,
    ) -> InferResult:
        """Run frame-by-frame inference and return an error result instead of raising."""
        hooks = hooks or InferHooks()
        start = perf_counter()
        output_dir = paths.INFERENCE_RUNS_DIR / "unknown" / "failed"
        log_path: Path | None = None
        try:
            config, merger = _build_service_config(yaml_path, cli_args)
            if config.source is None:
                raise ValueError("inference source is required")

            plan = build_inference_run_plan(config)
            output_dir = plan.source_run_dir
            run_logger = configure_run_logger(plan.log_file, logger_prefix="od_platform.inference")
            log_path = plan.log_file
            pipe_config = load_pipeline_config(pipeline_yaml)
            model_ref = resolve_model(config.model)
            source_ref = resolve_source(config.source)

            try:
                run_logger.info("frame pipeline inference run: %s", plan.audit_run_name)
                run_logger.info("model ref: %s", model_ref)
                run_logger.info("source ref: %s", source_ref)
                run_logger.info("output dir: %s", output_dir)
                run_logger.info("beautify: %s", beautify and pipe_config.viz_enabled)
                run_logger.info("threaded source: %s", threaded)
                _log_effective_config(config, run_logger, config_source=Path(yaml_path) if yaml_path else None)

                from ultralytics import YOLO

                model = YOLO(str(model_ref))
                names = _model_names(model)
                processor = _FrameProcessor(
                    model=model,
                    names=names,
                    predict_kwargs=_frame_predict_kwargs(config),
                    visualizer=_build_visualizer(names, pipe_config, enabled=beautify),
                    style_kwargs=pipe_config.normalized_style_overrides(),
                    use_label_mapping=pipe_config.use_label_mapping,
                    color_mapping=pipe_config.color_mapping,
                )
                sink = output_sink or (LocalFileSink() if config.save else NullSink())
                pipeline = SequentialInferencePipeline(
                    processor=processor,
                    source=source_ref,
                    camera_config=pipe_config.build_camera_config(),
                    output_dir=output_dir,
                    output_sink=sink,
                    save=config.save,
                    show=config.show,
                    show_info=show_info,
                    window_name=window_name,
                    warmup_frames=warmup_frames,
                    stride=config.vid_stride,
                    threaded=threaded,
                    hooks=hooks,
                    cancel_token=cancel_token,
                    pause_token=pause_token,
                    max_frames=max_frames,
                )
                stats = pipeline.run()
                audit_path = write_pipeline_audit(
                    plan,
                    config,
                    merger=merger.to_audit_log(),
                    pipeline_config=pipe_config,
                    source_ref=source_ref,
                    model_ref=model_ref,
                    stats=stats,
                )
                elapsed = perf_counter() - start
                result = InferResult(
                    success=True,
                    output_dir=output_dir,
                    stats=stats.to_dict(),
                    infer_time=elapsed,
                    saved=config.save,
                    audit_path=audit_path,
                    log_path=log_path,
                )
                run_logger.info("pipeline inference finished: frames=%s detections=%s", stats.frames, stats.detections)
                run_logger.info("audit: %s", audit_path)
                hooks.fire_complete(result)
                return result
            finally:
                _close_logger(run_logger)
        except Exception as exc:
            logger.exception("pipeline inference failed: %s", exc)
            hooks.fire_error(exc)
            return InferResult(
                success=False,
                output_dir=output_dir,
                stats={},
                infer_time=perf_counter() - start,
                error=str(exc),
                log_path=log_path,
            )


@dataclass
class _FrameProcessor:
    model: Any
    names: dict[int, str]
    predict_kwargs: dict[str, Any]
    visualizer: BeautifyVisualizer | None
    style_kwargs: dict[str, Any]
    use_label_mapping: bool
    color_mapping: dict[str, tuple[int, int, int]]
    _style: DrawStyle | None = None

    def infer(self, image: Any) -> tuple[Any, list[Detection]]:
        results = self.model(image, **self.predict_kwargs)
        result = results[0] if isinstance(results, (list, tuple)) else results
        detections = detections_from_yolo_result(result, names=self.names, color_mapping=self.color_mapping)
        return result, detections

    def draw(self, image: Any, result: Any, detections: list[Detection]) -> Any:
        if self.visualizer is None:
            return result.plot()
        if self._style is None:
            height, width = image.shape[:2]
            self._style = DrawStyle.from_image_size(height, width, **self.style_kwargs)
        return self.visualizer.draw(
            image,
            detections,
            style=self._style,
            use_label_mapping=self.use_label_mapping,
        )


def infer_yolo(
    yaml_path: str | Path | None = None,
    pipeline_yaml: str | Path | None = None,
    cli_args: dict[str, Any] | Namespace | None = None,
    *,
    beautify: bool = True,
    threaded: bool = False,
    warmup_frames: int = 0,
    window_name: str = "odp-infer",
    show_info: bool = True,
    output_sink: OutputSink | None = None,
    hooks: InferHooks | None = None,
    cancel_token: CancelToken | None = None,
    pause_token: PauseToken | None = None,
    max_frames: int | None = None,
) -> InferResult:
    """Convenience entry point parallel to the training service API."""
    return InferService().predict(
        yaml_path=yaml_path,
        pipeline_yaml=pipeline_yaml,
        cli_args=cli_args,
        beautify=beautify,
        threaded=threaded,
        warmup_frames=warmup_frames,
        window_name=window_name,
        show_info=show_info,
        output_sink=output_sink,
        hooks=hooks,
        cancel_token=cancel_token,
        pause_token=pause_token,
        max_frames=max_frames,
    )


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


def write_pipeline_audit(
    plan: InferenceRunPlan,
    config: YOLOInferConfig,
    *,
    merger: dict[str, Any],
    pipeline_config: PipelineConfig,
    source_ref: str | Path,
    model_ref: Path,
    stats: InferStats,
) -> Path:
    """Write D8 frame-pipeline audit metadata."""
    plan.source_run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": plan.source_run_name,
        "audit_run_name": plan.audit_run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": config.audit_snapshot(),
        "merger": merger,
        "pipeline": pipeline_config.to_audit(),
        "model_ref": str(model_ref),
        "source_ref": str(source_ref),
        "device_info": get_basic_device_info(),
        "outputs": {
            "run_dir": str(plan.source_run_dir),
            "log_file": str(plan.log_file),
        },
        "stats": stats.to_dict(),
    }
    output = plan.source_run_dir / "odp_audit.json"
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




def _log_effective_config(config: YOLOInferConfig, target_logger: logging.Logger, *, config_source: Path | None) -> None:
    target_logger.info("=" * 60)
    target_logger.info("effective inference config")
    source = str(config_source) if config_source else "default/direct"
    for field_name in config.__class__.model_fields:
        target_logger.info("%-20s: %s (source: %s)", field_name, getattr(config, field_name, None), source)
    target_logger.info("=" * 60)


def _build_service_config(
    yaml_path: str | Path | None,
    cli_args: dict[str, Any] | Namespace | None,
) -> tuple[YOLOInferConfig, Any]:
    if isinstance(cli_args, dict):
        cli_args = Namespace(**cli_args)
    config, merger = build_infer_config(yaml_path=yaml_path, cli_args=cli_args, dry_run=False)
    if config is None:
        raise RuntimeError("inference config preview unexpectedly returned None")
    return config, merger


def _frame_predict_kwargs(config: YOLOInferConfig) -> dict[str, Any]:
    keys = (
        "conf",
        "iou",
        "imgsz",
        "max_det",
        "classes",
        "agnostic_nms",
        "augment",
        "device",
        "retina_masks",
    )
    kwargs = {key: getattr(config, key) for key in keys if getattr(config, key, None) is not None}
    kwargs["verbose"] = False
    return kwargs


def _model_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def _build_visualizer(
    labels: dict[int, str],
    pipeline_config: PipelineConfig,
    *,
    enabled: bool,
) -> BeautifyVisualizer | None:
    if not enabled or not pipeline_config.viz_enabled:
        return None
    return BeautifyVisualizer(
        labels=[labels[key] for key in sorted(labels)],
        label_mapping=pipeline_config.label_mapping,
        color_mapping=pipeline_config.color_mapping,
        default_color=pipeline_config.default_color,
        font_path=pipeline_config.font_path,
    )


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
