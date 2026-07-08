"""Academic-style plots for Ultralytics training results."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from od_platform.training.metrics import read_results_csv

logger = logging.getLogger(__name__)


def plot_training_results(
    csv_path: Path,
    output_path: Path | None = None,
    *,
    title: str = "Object Detection Training Results",
    use_matplotx: bool = False,
) -> Path:
    """Render a 2x3 training-results figure from Ultralytics results.csv."""
    rows = read_results_csv(csv_path)
    epochs = _series(rows, "epoch")
    output = output_path or csv_path.with_name("training_results.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    if use_matplotx:
        logger.warning("matplotx style requested, but stable Pillow renderer is used in this environment")
    _plot_with_pillow(rows, epochs, output, title=title)
    logger.info("training results figure written: %s", output)
    return output


def _plot_with_pillow(rows: list[dict[str, float]], epochs: list[float], output: Path, *, title: str) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Pillow is required to plot training results") from exc

    width, height = 1500, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = font

    draw.text((width // 2 - 180, 24), title, fill=(20, 30, 45), font=title_font)
    panels = [
        (60, 90, 470, 350),
        (545, 90, 955, 350),
        (1030, 90, 1440, 350),
        (60, 460, 470, 720),
        (545, 460, 955, 720),
        (1030, 460, 1440, 720),
    ]
    palette = [
        (37, 99, 235),
        (220, 38, 38),
        (22, 163, 74),
        (147, 51, 234),
        (234, 88, 12),
        (8, 145, 178),
    ]
    _draw_panel(
        draw,
        panels[0],
        "Loss Curves",
        epochs,
        [
            ("Train Box", _series(rows, "train/box_loss"), palette[0]),
            ("Val Box", _series(rows, "val/box_loss"), palette[1]),
            ("Train Cls", _series(rows, "train/cls_loss"), palette[2]),
            ("Val Cls", _series(rows, "val/cls_loss"), palette[3]),
            ("Train DFL", _series(rows, "train/dfl_loss"), palette[4]),
            ("Val DFL", _series(rows, "val/dfl_loss"), palette[5]),
        ],
        font=font,
    )
    _draw_panel(
        draw,
        panels[1],
        "Total Loss",
        epochs,
        [
            ("Train Total", _total_loss_series(rows, "train"), palette[0]),
            ("Val Total", _total_loss_series(rows, "val"), palette[1]),
        ],
        font=font,
    )
    _draw_panel(
        draw,
        panels[2],
        "Evaluation Metrics",
        epochs,
        [
            ("Precision", _series(rows, "metrics/precision(B)"), palette[0]),
            ("Recall", _series(rows, "metrics/recall(B)"), palette[1]),
            ("mAP@50", _series(rows, "metrics/mAP50(B)"), palette[2]),
            ("mAP@50-95", _series(rows, "metrics/mAP50-95(B)"), palette[3]),
        ],
        y_min=0.0,
        y_max=1.05,
        font=font,
    )
    _draw_panel(
        draw,
        panels[3],
        "mAP Trend",
        epochs,
        [
            ("mAP@50", _series(rows, "metrics/mAP50(B)"), palette[2]),
            ("mAP@50-95", _series(rows, "metrics/mAP50-95(B)"), palette[3]),
        ],
        y_min=0.0,
        y_max=1.05,
        font=font,
    )
    _draw_panel(
        draw,
        panels[4],
        "Learning Rate",
        epochs,
        [
            ("lr/pg0", _series(rows, "lr/pg0"), palette[0]),
            ("lr/pg1", _series(rows, "lr/pg1"), palette[1]),
            ("lr/pg2", _series(rows, "lr/pg2"), palette[2]),
        ],
        font=font,
    )
    cumulative = _series(rows, "time")
    per_epoch = [cumulative[0], *[curr - prev for prev, curr in zip(cumulative, cumulative[1:], strict=False)]] if cumulative else []
    _draw_panel(
        draw,
        panels[5],
        "Epoch Time",
        epochs,
        [("Per Epoch", per_epoch, palette[0])],
        font=font,
    )
    image.save(output)


def _draw_panel(
    draw,
    box: tuple[int, int, int, int],
    title: str,
    x_values: list[float],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    *,
    y_min: float | None = None,
    y_max: float | None = None,
    font,
) -> None:
    left, top, right, bottom = box
    plot_left, plot_top, plot_right, plot_bottom = left + 48, top + 38, right - 18, bottom - 36
    draw.rectangle(box, outline=(203, 213, 225), width=1)
    draw.text((left + 12, top + 10), title, fill=(15, 23, 42), font=font)

    values = [value for _, ys, _ in series for value in ys if _is_finite(value)]
    if not x_values or not values:
        draw.text((plot_left, plot_top + 80), "No data", fill=(100, 116, 139), font=font)
        return

    x_min, x_max = min(x_values), max(x_values)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1.0
    actual_y_min = min(values) if y_min is None else y_min
    actual_y_max = max(values) if y_max is None else y_max
    if math.isclose(actual_y_min, actual_y_max):
        actual_y_max = actual_y_min + 1.0
    padding = (actual_y_max - actual_y_min) * 0.08 if y_min is None or y_max is None else 0.0
    actual_y_min -= padding
    actual_y_max += padding

    _draw_grid(draw, plot_left, plot_top, plot_right, plot_bottom)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(71, 85, 105), width=1)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(71, 85, 105), width=1)

    legend_x = left + 14
    legend_y = bottom - 25
    for index, (label, ys, color) in enumerate(series):
        points = _line_points(
            x_values,
            ys,
            plot_left=plot_left,
            plot_top=plot_top,
            plot_right=plot_right,
            plot_bottom=plot_bottom,
            x_min=x_min,
            x_max=x_max,
            y_min=actual_y_min,
            y_max=actual_y_max,
        )
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        lx = legend_x + (index % 3) * 128
        ly = legend_y + (index // 3) * 14
        draw.line((lx, ly + 6, lx + 16, ly + 6), fill=color, width=2)
        draw.text((lx + 20, ly), label, fill=(51, 65, 85), font=font)

    draw.text((plot_left, plot_bottom + 8), f"epoch {x_min:g}", fill=(100, 116, 139), font=font)
    draw.text((plot_right - 62, plot_bottom + 8), f"epoch {x_max:g}", fill=(100, 116, 139), font=font)
    draw.text((left + 8, plot_top - 4), f"{actual_y_max:.3g}", fill=(100, 116, 139), font=font)
    draw.text((left + 8, plot_bottom - 8), f"{actual_y_min:.3g}", fill=(100, 116, 139), font=font)


def _draw_grid(draw, left: int, top: int, right: int, bottom: int) -> None:
    for step in range(1, 4):
        x = left + (right - left) * step // 4
        y = top + (bottom - top) * step // 4
        draw.line((x, top, x, bottom), fill=(226, 232, 240), width=1)
        draw.line((left, y, right, y), fill=(226, 232, 240), width=1)


def _line_points(
    x_values: list[float],
    y_values: list[float],
    *,
    plot_left: int,
    plot_top: int,
    plot_right: int,
    plot_bottom: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for x_value, y_value in zip(x_values, y_values, strict=False):
        if not _is_finite(x_value) or not _is_finite(y_value):
            continue
        x_ratio = (x_value - x_min) / (x_max - x_min)
        y_ratio = (y_value - y_min) / (y_max - y_min)
        x = int(plot_left + x_ratio * (plot_right - plot_left))
        y = int(plot_bottom - y_ratio * (plot_bottom - plot_top))
        points.append((x, y))
    return points


def _is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _series(rows: list[dict[str, float]], key: str) -> list[float]:
    values = [row[key] for row in rows if key in row]
    return values if len(values) == len(rows) else []


def _total_loss_series(rows: list[dict[str, float]], prefix: str) -> list[float]:
    keys = [f"{prefix}/{name}_loss" for name in ("box", "cls", "dfl")]
    if any(any(key not in row for row in rows) for key in keys):
        return []
    return [sum(row[key] for key in keys) for row in rows]
