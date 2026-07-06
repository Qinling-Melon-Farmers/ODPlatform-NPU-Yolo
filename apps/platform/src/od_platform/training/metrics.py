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
    return "N/A"
