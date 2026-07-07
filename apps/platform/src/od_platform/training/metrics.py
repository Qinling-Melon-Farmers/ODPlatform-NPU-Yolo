"""Read and summarize Ultralytics training result CSV files."""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from od_platform.common.constants import Task
from od_platform.common.string_utils import pad_to_width

logger = logging.getLogger(__name__)

DETECT_METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


@dataclass(frozen=True)
class TrainMetrics:
    """Core metrics extracted from the last row of one results.csv."""

    task: str
    epoch: int
    fitness: float | None
    map50: float | None
    map50_95: float | None
    precision: float | None
    recall: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "epoch": self.epoch,
            "fitness": self.fitness,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
        }


def read_results_csv(csv_path: Path) -> list[dict[str, float]]:
    """Read Ultralytics results.csv into numeric row dictionaries."""
    if not csv_path.exists():
        raise FileNotFoundError(f"training results csv does not exist: {csv_path}")

    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                clean_key = key.strip()
                try:
                    row[clean_key] = float(str(value).strip())
                except (TypeError, ValueError):
                    row[clean_key] = math.nan
            rows.append(row)
    if not rows:
        raise ValueError(f"training results csv has no rows: {csv_path}")
    return rows


def summarize_results_csv(csv_path: Path, *, task: str = Task.DETECT) -> dict[str, Any]:
    """Create a compact summary for logs, manifests and reports."""
    rows = read_results_csv(csv_path)
    last = rows[-1]
    columns = DETECT_METRIC_COLUMNS
    metrics = TrainMetrics(
        task=task,
        epoch=int(_value(last, "epoch", 0) or 0),
        fitness=_fitness(last),
        map50=_value(last, columns["map50"]),
        map50_95=_value(last, columns["map50_95"]),
        precision=_value(last, columns["precision"]),
        recall=_value(last, columns["recall"]),
    )
    best_map50_row = _best_row(rows, columns["map50"])
    best_map95_row = _best_row(rows, columns["map50_95"])
    return {
        "csv_path": str(csv_path),
        "epochs": len(rows),
        "last": metrics.to_dict(),
        "best": {
            "map50": _row_metric(best_map50_row, columns["map50"]),
            "map50_95": _row_metric(best_map95_row, columns["map50_95"]),
        },
        "loss": {
            "last_train_total": _total_loss(last, "train"),
            "last_val_total": _total_loss(last, "val"),
        },
    }


def write_metrics_summary(csv_path: Path, output_path: Path, *, task: str = Task.DETECT) -> Path:
    """Write a JSON metrics summary next to training artifacts."""
    summary = summarize_results_csv(csv_path, task=task)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def log_metrics_summary(summary: dict[str, Any], *, target_logger: logging.Logger | None = None) -> None:
    """Log key metrics in a table-like format."""
    log = target_logger or logger
    last = summary.get("last", {})
    best = summary.get("best", {})
    log.info("training metrics summary: epochs=%s", summary.get("epochs"))
    log.info(
        "last metrics: precision=%s recall=%s map50=%s map50_95=%s",
        _fmt(last.get("precision")),
        _fmt(last.get("recall")),
        _fmt(last.get("map50")),
        _fmt(last.get("map50_95")),
    )
    log.info("best map50: %s", best.get("map50"))
    log.info("best map50_95: %s", best.get("map50_95"))


def log_training_report(
    summary: dict[str, Any],
    *,
    run_dir: Path | None = None,
    target_logger: logging.Logger | None = None,
    train_result: Any | None = None,
    width: int = 60,
) -> None:
    """Log a teaching-style training report from CSV summary and optional Ultralytics result."""
    log = target_logger or logger
    last = summary.get("last", {})
    best = summary.get("best", {})
    loss = summary.get("loss", {})
    result_metrics = _extract_result_metrics(train_result)
    task = result_metrics.get("task") or last.get("task") or "unknown"

    _log_section(log, f"训练结果 ({task})", width=width)
    _log_subtitle(log, "基本信息", width=width)
    _log_kv(log, "任务类型", task)
    _log_kv(log, "保存目录", run_dir or Path(str(summary.get("csv_path", ""))).parent)
    _log_kv(log, "训练轮数", summary.get("epochs"))

    speed = result_metrics.get("speed")
    if isinstance(speed, dict) and speed:
        _log_subtitle(log, "处理速度 (ms/image)", width=width)
        total = 0.0
        for key, label in (
            ("preprocess", "预处理"),
            ("inference", "推理"),
            ("loss", "损失计算"),
            ("postprocess", "后处理"),
        ):
            value = speed.get(key)
            if isinstance(value, (int, float)):
                total += float(value)
                _log_kv(log, label, f"{float(value):.3f} ms")
        _log_kv(log, "总计", f"{total:.3f} ms")

    _log_subtitle(log, "整体评估指标", width=width)
    _log_kv(log, "Fitness 分数", _metric_value(result_metrics, last, "fitness"))
    _log_kv(log, "Precision", _metric_value(result_metrics, last, "precision"))
    _log_kv(log, "Recall", _metric_value(result_metrics, last, "recall"))
    _log_kv(log, "mAP@50", _metric_value(result_metrics, last, "map50"))
    _log_kv(log, "mAP@50-95", _metric_value(result_metrics, last, "map50_95"))
    _log_kv(log, "最佳 mAP@50", _format_best_metric(best.get("map50")))
    _log_kv(log, "最佳 mAP@50-95", _format_best_metric(best.get("map50_95")))

    _log_subtitle(log, "损失指标", width=width)
    _log_kv(log, "最终训练总损失", loss.get("last_train_total"))
    _log_kv(log, "最终验证总损失", loss.get("last_val_total"))

    class_maps = result_metrics.get("class_maps")
    if isinstance(class_maps, list) and class_maps:
        _log_subtitle(log, "类别级 mAP@0.5:0.95 (Box)", width=width)
        for name, value in class_maps:
            _log_kv(log, str(name), value)
    log.info("=" * width)


def _value(row: dict[str, float], key: str, default: float | None = None) -> float | None:
    value = row.get(key, math.nan)
    if math.isnan(value):
        return default
    return value


def _fitness(row: dict[str, float]) -> float | None:
    map50 = _value(row, DETECT_METRIC_COLUMNS["map50"], 0.0) or 0.0
    map95 = _value(row, DETECT_METRIC_COLUMNS["map50_95"], 0.0) or 0.0
    return 0.1 * map50 + 0.9 * map95


def _total_loss(row: dict[str, float], prefix: str) -> float | None:
    values = [_value(row, f"{prefix}/{name}_loss") for name in ("box", "cls", "dfl")]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _best_row(rows: list[dict[str, float]], metric: str) -> dict[str, float]:
    return max(rows, key=lambda row: _value(row, metric, -math.inf) or -math.inf)


def _row_metric(row: dict[str, float], metric: str) -> dict[str, float | int | None]:
    return {"epoch": int(_value(row, "epoch", 0) or 0), "value": _value(row, metric)}


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return f"{float(value):.4f}"
    if value is not None:
        return str(value)
    return "N/A"


def _log_section(log: logging.Logger, title: str, *, width: int) -> None:
    log.info("=" * width)
    log.info(pad_to_width(title, width, "center"))
    log.info("=" * width)


def _log_subtitle(log: logging.Logger, title: str, *, width: int) -> None:
    log.info(pad_to_width(title, width, "center"))
    log.info("-" * width)


def _log_kv(log: logging.Logger, key: str, value: Any, *, key_width: int = 20) -> None:
    log.info("%s: %s", pad_to_width(key, key_width), _fmt(value))


def _format_best_metric(value: Any) -> str:
    if not isinstance(value, dict):
        return _fmt(value)
    metric = _fmt(value.get("value"))
    epoch = value.get("epoch")
    if epoch is None:
        return metric
    return f"{metric} (epoch {epoch})"


def _metric_value(result_metrics: dict[str, Any], last: dict[str, Any], key: str) -> Any:
    value = result_metrics.get(key)
    if value is not None:
        return value
    return last.get(key)


def _extract_result_metrics(train_result: Any | None) -> dict[str, Any]:
    if train_result is None:
        return {}

    metrics: dict[str, Any] = {}
    task = getattr(train_result, "task", None)
    if task is not None:
        metrics["task"] = task

    speed = getattr(train_result, "speed", None)
    if isinstance(speed, dict):
        metrics["speed"] = speed

    results_dict = getattr(train_result, "results_dict", None)
    if isinstance(results_dict, dict):
        metrics.update(
            {
                "precision": _first_present(results_dict, ("metrics/precision(B)", "precision")),
                "recall": _first_present(results_dict, ("metrics/recall(B)", "recall")),
                "map50": _first_present(results_dict, ("metrics/mAP50(B)", "map50")),
                "map50_95": _first_present(results_dict, ("metrics/mAP50-95(B)", "map50_95")),
                "fitness": _first_present(results_dict, ("fitness",)),
            }
        )

    class_maps = _extract_class_maps(train_result)
    if class_maps:
        metrics["class_maps"] = class_maps
    return metrics


def _first_present(values: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _extract_class_maps(train_result: Any) -> list[tuple[str, Any]]:
    names = getattr(train_result, "names", None)
    maps = getattr(train_result, "maps", None)
    if names is None or maps is None:
        return []

    try:
        map_values = list(maps)
    except TypeError:
        return []

    if isinstance(names, dict):
        return [(str(names.get(index, index)), value) for index, value in enumerate(map_values)]
    if isinstance(names, (list, tuple)):
        return [
            (str(names[index]) if index < len(names) else str(index), value)
            for index, value in enumerate(map_values)
        ]
    return []
