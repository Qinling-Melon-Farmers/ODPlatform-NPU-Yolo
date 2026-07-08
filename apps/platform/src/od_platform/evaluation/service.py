"""YOLO model evaluation service."""

from __future__ import annotations

import json
import logging
import re
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from od_platform.common import paths
from od_platform.common.refs import resolve_yaml
from od_platform.common.system_utils import get_basic_device_info
from od_platform.runtime_config.api import build_val_config
from od_platform.runtime_config.val import YOLOValConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValMetrics:
    """Core metrics from one validation run."""

    task: str
    fitness: float | None
    map50: float | None
    map50_95: float | None
    precision: float | None
    recall: float | None
    class_maps: dict[str, float] | None = None
    speed: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "fitness": self.fitness,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
            "class_maps": self.class_maps or {},
            "speed": self.speed or {},
        }


@dataclass(frozen=True)
class EvaluationRunPlan:
    """Filesystem layout for one evaluation run."""

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
class ValResult:
    """Result metadata returned by model evaluation."""

    success: bool
    plan: EvaluationRunPlan | None
    save_dir: Path | None
    metrics: ValMetrics | None
    audit_path: Path | None = None
    summary_path: Path | None = None
    error: str | None = None


class ValService:
    """Orchestrate runtime config, trained weight resolution and YOLO validation."""

    def evaluate(
        self,
        config_path: str | Path | None = "val",
        model: str | None = None,
        data: str | None = None,
        *,
        cli_overrides: dict[str, Any] | Namespace | None = None,
        executor: str | None = None,
    ) -> ValResult:
        """Run one validation and return a result instead of raising."""
        start = perf_counter()
        plan: EvaluationRunPlan | None = None
        try:
            cli_args = _build_cli_namespace(model=model, data=data, cli_overrides=cli_overrides)
            config, merger = build_val_config(config_path, cli_args, dry_run=False)
            if config is None:
                raise RuntimeError("validation config preview unexpectedly returned None")

            weight = _resolve_existing_model(config.model)
            data_yaml = resolve_yaml(config.data)
            if not data_yaml.exists():
                raise FileNotFoundError(f"dataset yaml does not exist: {data_yaml}")

            plan = build_evaluation_run_plan(config)
            logger.info("evaluation run: %s", plan.audit_run_name)
            logger.info("model ref: %s", weight)
            logger.info("dataset yaml: %s", data_yaml)
            logger.info("output dir: %s", plan.source_run_dir)

            raw_result = self._run_eval(weight, config, data_yaml, plan)
            elapsed_seconds = perf_counter() - start
            save_dir = _extract_save_dir(raw_result) or plan.source_run_dir
            metrics = _extract_metrics(raw_result, config.task)
            summary_path = write_evaluation_summary(save_dir, metrics, elapsed_seconds=elapsed_seconds)
            audit_path = write_evaluation_audit(
                plan,
                config,
                merger=merger.to_audit_log(),
                model_ref=weight,
                data_yaml=data_yaml,
                metrics=metrics,
                save_dir=save_dir,
                elapsed_seconds=elapsed_seconds,
                executor=executor,
            )
            _log_metrics(metrics, elapsed_seconds=elapsed_seconds)
            logger.info("evaluation audit: %s", audit_path)
            return ValResult(
                success=True,
                plan=plan,
                save_dir=save_dir,
                metrics=metrics,
                audit_path=audit_path,
                summary_path=summary_path,
            )
        except Exception as exc:
            logger.exception("model evaluation failed: %s", exc)
            return ValResult(
                success=False,
                plan=plan,
                save_dir=plan.source_run_dir if plan is not None else None,
                metrics=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _run_eval(weight: Path, config: YOLOValConfig, data_yaml: Path, plan: EvaluationRunPlan) -> Any:
        from ultralytics import YOLO

        kwargs = config.to_ultralytics_kwargs()
        kwargs.update(
            {
                "data": str(data_yaml),
                "project": str(plan.ultralytics_project),
                "name": plan.source_run_name,
            }
        )
        model = YOLO(str(weight))
        return model.val(**kwargs)


def build_evaluation_run_plan(config: YOLOValConfig, *, now: datetime | None = None) -> EvaluationRunPlan:
    """Build output and log names for one validation run."""
    sequence = _next_evaluation_sequence(config.task)
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    model_slug = _model_slug(config.model)
    source_run_name = config.name or f"val-{sequence}"
    audit_run_name = f"{source_run_name}-{timestamp}-{model_slug}"
    ultralytics_project = Path(config.project).resolve() if config.project else paths.RUNS_DIR / "evaluation" / config.task
    return EvaluationRunPlan(
        sequence=sequence,
        timestamp=timestamp,
        task=config.task,
        model_slug=model_slug,
        source_run_name=source_run_name,
        audit_run_name=audit_run_name,
        source_run_dir=ultralytics_project / source_run_name,
        ultralytics_project=ultralytics_project,
        log_file=paths.LOGGING_DIR / "evaluation" / f"{audit_run_name}.log",
    )


def evaluate_yolo(
    config_path: str | Path | None = "val",
    model: str | None = None,
    data: str | None = None,
    *,
    cli_overrides: dict[str, Any] | Namespace | None = None,
    executor: str | None = None,
) -> ValResult:
    """Convenience entry point for model evaluation."""
    return ValService().evaluate(
        config_path=config_path,
        model=model,
        data=data,
        cli_overrides=cli_overrides,
        executor=executor,
    )


def write_evaluation_summary(save_dir: Path, metrics: ValMetrics, *, elapsed_seconds: float) -> Path:
    """Write a compact JSON summary next to validation outputs."""
    save_dir.mkdir(parents=True, exist_ok=True)
    output = save_dir / "evaluation_summary.json"
    payload = {"elapsed_seconds": round(elapsed_seconds, 6), "metrics": metrics.to_dict()}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def write_evaluation_audit(
    plan: EvaluationRunPlan,
    config: YOLOValConfig,
    *,
    merger: dict[str, Any],
    model_ref: Path,
    data_yaml: Path,
    metrics: ValMetrics,
    save_dir: Path,
    elapsed_seconds: float,
    executor: str | None,
) -> Path:
    """Write evaluation audit metadata."""
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "val",
        "run_name": plan.source_run_name,
        "audit_run_name": plan.audit_run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "executor": executor,
        "config": config.audit_snapshot(),
        "merger": merger,
        "model_ref": str(model_ref),
        "data_yaml": str(data_yaml),
        "device_info": get_basic_device_info(),
        "outputs": {
            "run_dir": str(save_dir),
            "log_file": str(plan.log_file),
        },
        "elapsed_seconds": round(elapsed_seconds, 6),
        "metrics": metrics.to_dict(),
    }
    output = save_dir / "odp_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _build_cli_namespace(
    *,
    model: str | None,
    data: str | None,
    cli_overrides: dict[str, Any] | Namespace | None,
) -> Namespace:
    if cli_overrides is None:
        payload: dict[str, Any] = {}
    elif isinstance(cli_overrides, Namespace):
        payload = vars(cli_overrides).copy()
    else:
        payload = dict(cli_overrides)
    if model is not None:
        payload["model"] = model
    if data is not None:
        payload["data"] = data
    return Namespace(**payload)


def _resolve_existing_model(ref: str) -> Path:
    """Resolve an evaluation model and require it to exist locally."""
    path = Path(ref)
    if path.is_absolute() or len(path.parts) > 1:
        candidates = [path, paths.ROOT_DIR / path]
    else:
        candidates = [
            paths.CHECKPOINTS_DIR / path.name,
            paths.TRAINED_MODELS_DIR / path.name,
            paths.ROOT_DIR / path.name,
        ]
        candidates.extend(sorted(paths.TRAINED_MODELS_DIR.glob(f"**/{path.name}")))
        candidates.extend(sorted(paths.CHECKPOINTS_DIR.glob(f"**/{path.name}")))

    resolved: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir():
            for child in ("best.pt", "last.pt"):
                if (candidate / child).exists():
                    resolved.append((candidate / child).resolve())
        elif candidate.exists():
            resolved.append(candidate)

    unique = list(dict.fromkeys(resolved))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        options = ", ".join(str(item) for item in unique[:5])
        raise ValueError(f"model reference is ambiguous: {ref}. Candidates: {options}")
    raise FileNotFoundError(f"trained model does not exist: {ref}")


def _extract_save_dir(results: Any) -> Path | None:
    for value in (
        getattr(results, "save_dir", None),
        getattr(getattr(results, "validator", None), "save_dir", None),
        getattr(getattr(results, "args", None), "save_dir", None),
    ):
        if value is not None:
            return Path(value).resolve()
    return None


def _extract_metrics(results: Any, task: str) -> ValMetrics:
    result_dict = getattr(results, "results_dict", None)
    if not isinstance(result_dict, dict):
        result_dict = {}
    box = getattr(results, "box", None)
    return ValMetrics(
        task=task,
        fitness=_metric(results, result_dict, "fitness", "fitness"),
        map50=_metric(box, result_dict, "map50", "metrics/mAP50(B)"),
        map50_95=_metric(box, result_dict, "map", "metrics/mAP50-95(B)"),
        precision=_metric(box, result_dict, "mp", "metrics/precision(B)"),
        recall=_metric(box, result_dict, "mr", "metrics/recall(B)"),
        class_maps=_extract_class_maps(results),
        speed=_json_float_dict(getattr(results, "speed", None)),
    )


def _metric(obj: Any, data: dict[str, Any], attr: str, key: str) -> float | None:
    value = getattr(obj, attr, None) if obj is not None else None
    if value is None:
        value = data.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_class_maps(results: Any) -> dict[str, float]:
    names = getattr(results, "names", None)
    maps = getattr(results, "maps", None)
    if names is None or maps is None:
        return {}
    try:
        values = list(maps)
    except TypeError:
        return {}
    output: dict[str, float] = {}
    for index, value in enumerate(values):
        if isinstance(names, dict):
            label = str(names.get(index, index))
        elif isinstance(names, (list, tuple)) and index < len(names):
            label = str(names[index])
        else:
            label = str(index)
        try:
            output[label] = float(value)
        except (TypeError, ValueError):
            continue
    return output


def _json_float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, float] = {}
    for key, item in value.items():
        try:
            output[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return output


def _log_metrics(metrics: ValMetrics, *, elapsed_seconds: float) -> None:
    logger.info("evaluation finished in %.3f seconds", elapsed_seconds)
    logger.info(
        "metrics: fitness=%s precision=%s recall=%s map50=%s map50_95=%s",
        _fmt(metrics.fitness),
        _fmt(metrics.precision),
        _fmt(metrics.recall),
        _fmt(metrics.map50),
        _fmt(metrics.map50_95),
    )
    if metrics.class_maps:
        logger.info("class maps: %s", json.dumps(metrics.class_maps, ensure_ascii=False, sort_keys=True))


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _next_evaluation_sequence(task: str) -> int:
    pattern = re.compile(r"^val-(\d+)(?:\D.*)?$")
    base = paths.RUNS_DIR / "evaluation" / task
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
