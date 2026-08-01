"""Training service with auditable logs and model archiving."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from od_platform.common import paths
from od_platform.common.logging_utils import configure_run_logger
from od_platform.common.refs import resolve_model, resolve_yaml
from od_platform.common.string_utils import model_slug, pad_to_width
from od_platform.common.system_utils import get_basic_device_info
from od_platform.runtime_config.train import YOLOTrainConfig
from od_platform.training.metrics import (
    log_training_report,
    summarize_results_csv,
    write_metrics_summary,
)
from od_platform.training.plots import plot_training_results

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRunPlan:
    """All filesystem names derived for one training run."""

    sequence: int
    timestamp: str
    task: str
    model_slug: str
    source_run_name: str
    archive_run_name: str
    source_run_dir: Path
    ultralytics_project: Path
    log_file: Path
    archive_root: Path


@dataclass(frozen=True)
class TrainingRunResult:
    """Result metadata returned by run_training."""

    plan: TrainingRunPlan
    dry_run: bool
    manifest_path: Path
    archived_weights: dict[str, Path]


def build_training_run_plan(config: YOLOTrainConfig, *, now: datetime | None = None) -> TrainingRunPlan:
    """Build a stable run plan: train-N, log file, and archive base name."""
    sequence = _next_training_sequence(config.task)
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    model_slug_value = model_slug(config.model)
    ultralytics_project = Path(config.project).resolve() if config.project else paths.RUNS_DIR / config.task
    requested_run_name = config.name or f"train-{sequence}"
    source_run_name = _available_run_name(ultralytics_project, requested_run_name, exist_ok=config.exist_ok)
    archive_run_name = f"{source_run_name}-{timestamp}-{model_slug_value}"
    return TrainingRunPlan(
        sequence=sequence,
        timestamp=timestamp,
        task=config.task,
        model_slug=model_slug_value,
        source_run_name=source_run_name,
        archive_run_name=archive_run_name,
        source_run_dir=ultralytics_project / source_run_name,
        ultralytics_project=ultralytics_project,
        log_file=paths.LOGGING_DIR / "training" / f"{archive_run_name}.log",
        archive_root=paths.TRAINED_MODELS_DIR,
    )


def run_training(
    config: YOLOTrainConfig,
    *,
    config_source: Path | None = None,
    executor: str | None = None,
    dry_run: bool = False,
) -> TrainingRunResult:
    """Run or dry-run YOLO training with audit logs and post-run archiving."""
    plan = build_training_run_plan(config)
    run_logger = configure_run_logger(plan.log_file, logger_prefix="od_platform.training")
    dataset_yaml = resolve_yaml(config.data)
    model_ref = resolve_model(config.model)
    dataset_summary = summarize_dataset(dataset_yaml)

    run_logger.info("training run: %s", plan.archive_run_name)
    run_logger.info("executor: %s", executor or "unknown")
    run_logger.info("config source: %s", config_source or "direct object")
    run_logger.info("dataset yaml: %s", dataset_yaml)
    run_logger.info("model ref: %s", model_ref)
    run_logger.info("dataset summary: %s", json.dumps(dataset_summary, ensure_ascii=False, sort_keys=True))
    run_logger.info("ultralytics output: %s", plan.source_run_dir)
    run_logger.info("model archive root: %s", plan.archive_root)
    _log_effective_config(config, run_logger, config_source=config_source)

    manifest_path: Path | None = None
    if dry_run:
        manifest_path = write_training_manifest(
            plan,
            config,
            config_source=config_source,
            executor=executor,
            dataset_summary=dataset_summary,
            dry_run=True,
        )

    archived: dict[str, Path] = {}
    if not dry_run:
        kwargs = config.to_ultralytics_kwargs()
        kwargs.update(
            {
                "data": str(dataset_yaml),
                "project": str(plan.ultralytics_project),
                "name": plan.source_run_name,
            }
        )
        run_logger.info("start ultralytics training")
        from ultralytics import YOLO

        model = YOLO(str(model_ref))
        start_time = perf_counter()
        train_result = model.train(**kwargs)
        elapsed_seconds = perf_counter() - start_time
        actual_run_dir = _resolve_completed_run_dir(plan.source_run_dir, train_result)
        if actual_run_dir != plan.source_run_dir:
            run_logger.warning("ultralytics output dir changed: %s -> %s", plan.source_run_dir, actual_run_dir)
            plan = replace(plan, source_run_name=actual_run_dir.name, source_run_dir=actual_run_dir)
        run_logger.info("ultralytics training finished")
        training_outputs = inspect_training_outputs(plan.source_run_dir, task=config.task, target_logger=run_logger)
        if config.archive_weights:
            archived = archive_model_weights(
                plan.source_run_dir,
                plan.archive_run_name,
                config.model,
                copy=config.copy_archive,
            )
            run_logger.info("archived weights: %s", {key: str(value) for key, value in archived.items()})
        training_outputs["elapsed_seconds"] = round(elapsed_seconds, 3)
        training_outputs["archived_weights"] = {key: str(value) for key, value in archived.items()}
        metrics_summary = training_outputs.get("metrics_summary")
        if isinstance(metrics_summary, dict):
            log_training_report(
                metrics_summary,
                run_dir=plan.source_run_dir,
                target_logger=run_logger,
                train_result=train_result,
            )
        _log_training_outputs(plan, elapsed_seconds, archived, run_logger)
        manifest_path = write_training_manifest(
            plan,
            config,
            config_source=config_source,
            executor=executor,
            dataset_summary=dataset_summary,
            dry_run=False,
            training_outputs=training_outputs,
        )
    else:
        run_logger.info("dry run enabled; ultralytics training skipped")

    _close_logger(run_logger)
    if manifest_path is None:
        raise RuntimeError("training manifest was not written")
    return TrainingRunResult(plan=plan, dry_run=dry_run, manifest_path=manifest_path, archived_weights=archived)


def write_training_manifest(
    plan: TrainingRunPlan,
    config: YOLOTrainConfig,
    *,
    config_source: Path | None,
    executor: str | None,
    dataset_summary: dict[str, Any] | None = None,
    dry_run: bool = False,
    training_outputs: dict[str, Any] | None = None,
) -> Path:
    """Write a machine-readable training manifest next to the run output."""
    plan.source_run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": plan.source_run_name,
        "archive_run_name": plan.archive_run_name,
        "dry_run": dry_run,
        "executor": executor,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_source": str(config_source) if config_source else None,
        "config": config.audit_snapshot(),
        "dataset": dataset_summary or summarize_dataset(resolve_yaml(config.data)),
        "device_info": get_basic_device_info(),
        "outputs": {
            "ultralytics_run_dir": str(plan.source_run_dir),
            "log_file": str(plan.log_file),
            "archive_root": str(plan.archive_root),
        },
        "training_outputs": training_outputs or inspect_training_outputs(plan.source_run_dir, task=config.task),
    }
    output = plan.source_run_dir / "training_manifest.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def inspect_training_outputs(
    run_dir: Path,
    *,
    task: str,
    target_logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Inspect results.csv and produce summary/figure files when available."""
    results_csv = run_dir / "results.csv"
    output: dict[str, Any] = {"results_csv": str(results_csv), "results_csv_exists": results_csv.exists()}
    if not results_csv.exists():
        return output

    try:
        summary = summarize_results_csv(results_csv, task=task)
        summary_path = write_metrics_summary(results_csv, run_dir / "training_metrics.json", task=task)
        output["metrics_summary"] = summary
        output["metrics_summary_path"] = str(summary_path)
    except Exception as exc:
        output["metrics_error"] = f"{type(exc).__name__}: {exc}"
        if target_logger is not None:
            target_logger.warning("training metrics summary failed: %s", exc)

    try:
        figure_path = plot_training_results(results_csv, run_dir / "training_results.png")
        output["figure_path"] = str(figure_path)
    except Exception as exc:
        output["figure_error"] = f"{type(exc).__name__}: {exc}"
        if target_logger is not None:
            target_logger.warning("training plot failed: %s", exc)
    return output


def summarize_dataset(yaml_path: Path) -> dict[str, Any]:
    """Read basic dataset information for training logs and manifests."""
    summary: dict[str, Any] = {"yaml_path": str(yaml_path), "exists": yaml_path.exists()}
    if not yaml_path.exists():
        return summary
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        summary["load_error"] = f"{type(exc).__name__}: {exc}"
        return summary
    if not isinstance(data, dict):
        summary["load_error"] = f"top-level is {type(data).__name__}, expected dict"
        return summary
    summary.update(
        {
            "path": data.get("path"),
            "train": data.get("train"),
            "val": data.get("val"),
            "test": data.get("test"),
            "nc": data.get("nc"),
            "names": data.get("names"),
        }
    )
    return summary


def archive_model_weights(
    source_run_dir: Path,
    archive_run_name: str,
    model_name: str,
    *,
    copy: bool = True,
) -> dict[str, Path]:
    """Archive Ultralytics best.pt and last.pt into models/trained."""
    archived: dict[str, Path] = {}
    weights_dir = source_run_dir / "weights"
    for kind in ("best", "last"):
        src = weights_dir / f"{kind}.pt"
        if not src.exists():
            logger.warning("weight file not found, skip archive: %s", src)
            continue
        dst_dir = paths.TRAINED_MODELS_DIR / f"{archive_run_name}-{kind}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{kind}.pt"
        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        metadata = {
            "kind": kind,
            "model": model_name,
            "source": str(src),
            "archive": str(dst),
            "archived_at": datetime.now().isoformat(timespec="seconds"),
        }
        (dst_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        archived[kind] = dst
    return archived


def _next_training_sequence(task: str) -> int:
    pattern = re.compile(r"^train-(\d+)(?:\D.*)?$")
    numbers: list[int] = []
    for base in (paths.RUNS_DIR / task, paths.TRAINED_MODELS_DIR):
        if not base.exists():
            continue
        for item in base.iterdir():
            match = pattern.match(item.name)
            if match:
                numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _available_run_name(project: Path, requested_name: str, *, exist_ok: bool) -> str:
    """Return a run name that will not be auto-renamed by Ultralytics."""
    if exist_ok or not (project / requested_name).exists():
        return requested_name
    index = 2
    while (project / f"{requested_name}-{index}").exists():
        index += 1
    return f"{requested_name}-{index}"


def _resolve_completed_run_dir(planned_dir: Path, train_result: Any) -> Path:
    """Resolve the actual Ultralytics output directory after training."""
    for candidate in _result_run_dir_candidates(train_result):
        if _has_training_artifacts(candidate):
            return candidate
    if _has_training_artifacts(planned_dir):
        return planned_dir
    siblings = sorted(
        planned_dir.parent.glob(f"{planned_dir.name}*"),
        key=lambda item: item.stat().st_mtime if item.exists() else 0,
        reverse=True,
    )
    for candidate in siblings:
        if candidate.is_dir() and _has_training_artifacts(candidate):
            return candidate
    return planned_dir


def _result_run_dir_candidates(train_result: Any) -> list[Path]:
    candidates: list[Path] = []
    for value in (
        getattr(train_result, "save_dir", None),
        getattr(getattr(train_result, "trainer", None), "save_dir", None),
        getattr(getattr(train_result, "args", None), "save_dir", None),
    ):
        if value is not None:
            candidates.append(Path(value).resolve())
    return candidates


def _has_training_artifacts(run_dir: Path) -> bool:
    return (run_dir / "results.csv").exists() or (run_dir / "weights" / "best.pt").exists()


def _log_effective_config(config: YOLOTrainConfig, target_logger: logging.Logger, *, config_source: Path | None) -> None:
    target_logger.info("=" * 60)
    target_logger.info("effective training config")
    source = str(config_source) if config_source else "default/direct"
    for field_name in config.__class__.model_fields:
        target_logger.info("%-20s: %s (source: %s)", field_name, getattr(config, field_name, None), source)
    target_logger.info("=" * 60)


def _log_training_outputs(
    plan: TrainingRunPlan,
    elapsed_seconds: float,
    archived_weights: dict[str, Path],
    target_logger: logging.Logger,
) -> None:
    width = 60
    target_logger.info("=" * width)
    target_logger.info(pad_to_width("训练产物", width, "center"))
    target_logger.info("-" * width)
    target_logger.info("%s: %.2f 秒", pad_to_width("训练总耗时", 20), elapsed_seconds)
    target_logger.info("%s: %s", pad_to_width("输出目录", 20), plan.source_run_dir)
    target_logger.info("%s: %s", pad_to_width("审计清单", 20), plan.source_run_dir / "training_manifest.json")
    best = archived_weights.get("best") or plan.source_run_dir / "weights" / "best.pt"
    target_logger.info("%s: %s", pad_to_width("最佳权重", 20), best)
    for kind, path in archived_weights.items():
        target_logger.info("%s: %s", pad_to_width(f"{kind} 归档", 20), path)
    target_logger.info("=" * width)


def _close_logger(run_logger: logging.Logger) -> None:
    for handler in list(run_logger.handlers):
        handler.close()
        run_logger.removeHandler(handler)
