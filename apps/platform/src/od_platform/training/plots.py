"""Academic-style plots for Ultralytics training results."""

from __future__ import annotations

import logging
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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot training results") from exc

    _apply_style(plt, use_matplotx=use_matplotx)

    epochs = _series(rows, "epoch")
    output = output_path or csv_path.with_name("training_results.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    _plot_loss_panel(axes[0][0], epochs, rows)
    _plot_total_loss_panel(axes[0][1], epochs, rows)
    _plot_metrics_panel(axes[0][2], epochs, rows)
    _plot_map_panel(axes[1][0], epochs, rows)
    _plot_lr_panel(axes[1][1], epochs, rows)
    _plot_time_panel(axes[1][2], epochs, rows)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("training results figure written: %s", output)
    return output


def _apply_style(plt, *, use_matplotx: bool) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "axes.unicode_minus": False,
        }
    )
    if not use_matplotx:
        return
    try:
        import matplotx

        style = getattr(matplotx.styles, "pitaya_smoothie", None)
        if isinstance(style, dict) and "light" in style:
            plt.style.use(style["light"])
    except Exception as exc:
        logger.warning("matplotx style unavailable, fallback to matplotlib defaults: %s", exc)


def _plot_loss_panel(ax, epochs: list[float], rows: list[dict[str, float]]) -> None:
    for key, label in (
        ("box_loss", "Box"),
        ("cls_loss", "Cls"),
        ("dfl_loss", "DFL"),
    ):
        _safe_plot(ax, epochs, rows, f"train/{key}", label=f"Train {label}")
        _safe_plot(ax, epochs, rows, f"val/{key}", label=f"Val {label}", linestyle="--")
    _finish(ax, "Loss Curves", "Epoch", "Loss")


def _plot_total_loss_panel(ax, epochs: list[float], rows: list[dict[str, float]]) -> None:
    train_total = _total_loss_series(rows, "train")
    val_total = _total_loss_series(rows, "val")
    if train_total:
        ax.plot(epochs, train_total, label="Train Total")
    if val_total:
        ax.plot(epochs, val_total, label="Val Total", linestyle="--")
    _finish(ax, "Total Loss", "Epoch", "Loss")


def _plot_metrics_panel(ax, epochs: list[float], rows: list[dict[str, float]]) -> None:
    for key, label in (
        ("metrics/precision(B)", "Precision"),
        ("metrics/recall(B)", "Recall"),
        ("metrics/mAP50(B)", "mAP@50"),
        ("metrics/mAP50-95(B)", "mAP@50-95"),
    ):
        _safe_plot(ax, epochs, rows, key, label=label)
    ax.set_ylim(0, 1.05)
    _finish(ax, "Evaluation Metrics", "Epoch", "Score")


def _plot_map_panel(ax, epochs: list[float], rows: list[dict[str, float]]) -> None:
    for key, label in (
        ("metrics/mAP50(B)", "mAP@50"),
        ("metrics/mAP50-95(B)", "mAP@50-95"),
    ):
        values = _series(rows, key)
        if values:
            ax.plot(epochs, values, label=label)
            ax.axhline(sum(values) / len(values), linestyle=":", linewidth=1, alpha=0.7)
    ax.set_ylim(0, 1.05)
    _finish(ax, "mAP Trend", "Epoch", "mAP")


def _plot_lr_panel(ax, epochs: list[float], rows: list[dict[str, float]]) -> None:
    plotted = False
    for key in ("lr/pg0", "lr/pg1", "lr/pg2"):
        plotted = _safe_plot(ax, epochs, rows, key, label=key) or plotted
    if plotted:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    _finish(ax, "Learning Rate", "Epoch", "LR")


def _plot_time_panel(ax, epochs: list[float], rows: list[dict[str, float]]) -> None:
    cumulative = _series(rows, "time")
    if cumulative:
        per_epoch = [cumulative[0], *[curr - prev for prev, curr in zip(cumulative, cumulative[1:], strict=False)]]
        ax.plot(epochs, per_epoch, marker=".", markersize=2, label="Per Epoch")
    _finish(ax, "Epoch Time", "Epoch", "Seconds")


def _safe_plot(ax, epochs: list[float], rows: list[dict[str, float]], key: str, **kwargs) -> bool:
    values = _series(rows, key)
    if not values:
        return False
    ax.plot(epochs, values, **kwargs)
    return True


def _series(rows: list[dict[str, float]], key: str) -> list[float]:
    values = [row[key] for row in rows if key in row]
    return values if len(values) == len(rows) else []


def _total_loss_series(rows: list[dict[str, float]], prefix: str) -> list[float]:
    keys = [f"{prefix}/{name}_loss" for name in ("box", "cls", "dfl")]
    if any(any(key not in row for row in rows) for key in keys):
        return []
    return [sum(row[key] for key in keys) for row in rows]


def _finish(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(fontsize=8)
