"""模型目录：内置 CV 模型元数据库。

设计目标：对任务类型 (detect/classify/segment) 与模型后端
(ultralytics/torchvision) 保持开放，便于未来引入分类、分割等
更多 CV 模型。内置目录按 ``(task, family)`` 组织，指标统一存入
``metrics`` 字典，主指标键由 ``PRIMARY_METRIC`` 按任务映射，
未来新任务只需补充映射表与模型条目。

@FileName:   catalog.py
@Function:   内置模型元数据、按任务/系列查询
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

from od_platform.common.constants import Task

logger = logging.getLogger(__name__)

#: 各任务的主指标键。新任务（如 classify -> top1_acc）在此扩展。
PRIMARY_METRIC: Final[dict[str, str]] = {
    Task.DETECT: "map50_95",
}


@dataclass(frozen=True)
class ModelInfo:
    """一个模型的元数据条目。

    Attributes:
        name:          模型引用名，如 ``yolo11n.pt``、``resnet18``。
        family:        模型系列，如 ``yolo11``、``yolov8``、``resnet``。
        variant:       系列内变体标识，如 ``n``、``18``。
        task:          任务类型，见 ``common.constants.Task``。
        backend:       运行后端，如 ``ultralytics``、``torchvision``。
        size_category: 尺寸档位，如 nano/small/medium/large/xlarge。
        params_m:      参数量（百万）。
        metrics:       任务主指标字典，如 ``{"map50_95": 39.5}``。
        speed_cpu_ms:  CPU 推理耗时（毫秒，近似参考值）。
        description:   中文描述，供推荐与展示。
    """

    name: str
    family: str
    variant: str
    task: str
    backend: str
    size_category: str
    params_m: float
    metrics: dict[str, float] = field(default_factory=dict)
    speed_cpu_ms: float = 0.0
    description: str = ""

    @property
    def primary_metric(self) -> float | None:
        """返回当前任务主指标值，未知任务返回 None。"""
        key = PRIMARY_METRIC.get(self.task)
        if key is None:
            return None
        return self.metrics.get(key)


#: 内置模型目录（有序）。用户扩展模型由 loader.py 合并进总目录。
BUILTIN_MODELS: Final[dict[str, ModelInfo]] = {
    # ---- YOLO11 检测系列 ----
    "yolo11n.pt": ModelInfo(
        name="yolo11n.pt",
        family="yolo11",
        variant="n",
        task=Task.DETECT,
        backend="ultralytics",
        size_category="nano",
        params_m=2.6,
        metrics={"map50_95": 39.5},
        speed_cpu_ms=12.0,
        description="YOLO11 Nano：最快速度，适合实时检测",
    ),
    "yolo11s.pt": ModelInfo(
        name="yolo11s.pt",
        family="yolo11",
        variant="s",
        task=Task.DETECT,
        backend="ultralytics",
        size_category="small",
        params_m=9.4,
        metrics={"map50_95": 47.0},
        speed_cpu_ms=25.0,
        description="YOLO11 Small：速度与精度平衡",
    ),
    "yolo11m.pt": ModelInfo(
        name="yolo11m.pt",
        family="yolo11",
        variant="m",
        task=Task.DETECT,
        backend="ultralytics",
        size_category="medium",
        params_m=20.1,
        metrics={"map50_95": 51.5},
        speed_cpu_ms=50.0,
        description="YOLO11 Medium：精度优先的均衡选择",
    ),
    "yolo11l.pt": ModelInfo(
        name="yolo11l.pt",
        family="yolo11",
        variant="l",
        task=Task.DETECT,
        backend="ultralytics",
        size_category="large",
        params_m=25.3,
        metrics={"map50_95": 53.4},
        speed_cpu_ms=70.0,
        description="YOLO11 Large：高精度，适合离线高要求场景",
    ),
    "yolo11x.pt": ModelInfo(
        name="yolo11x.pt",
        family="yolo11",
        variant="x",
        task=Task.DETECT,
        backend="ultralytics",
        size_category="xlarge",
        params_m=56.9,
        metrics={"map50_95": 54.7},
        speed_cpu_ms=120.0,
        description="YOLO11 XLarge：最高精度，计算开销最大",
    ),
    # ---- YOLOv8 检测系列 ----
    "yolov8n.pt": ModelInfo(
        name="yolov8n.pt",
        family="yolov8",
        variant="n",
        task=Task.DETECT,
        backend="ultralytics",
        size_category="nano",
        params_m=3.2,
        metrics={"map50_95": 37.3},
        speed_cpu_ms=13.0,
        description="YOLOv8 Nano：轻量快速，成熟稳定",
    ),
    "yolov8s.pt": ModelInfo(
        name="yolov8s.pt",
        family="yolov8",
        variant="s",
        task=Task.DETECT,
        backend="ultralytics",
        size_category="small",
        params_m=11.2,
        metrics={"map50_95": 44.9},
        speed_cpu_ms=28.0,
        description="YOLOv8 Small：速度与精度平衡",
    ),
    "yolov8m.pt": ModelInfo(
        name="yolov8m.pt",
        family="yolov8",
        variant="m",
        task=Task.DETECT,
        backend="ultralytics",
        size_category="medium",
        params_m=25.9,
        metrics={"map50_95": 50.2},
        speed_cpu_ms=55.0,
        description="YOLOv8 Medium：中等规模常用选择",
    ),
    "yolov8l.pt": ModelInfo(
        name="yolov8l.pt",
        family="yolov8",
        variant="l",
        task=Task.DETECT,
        backend="ultralytics",
        size_category="large",
        params_m=43.7,
        metrics={"map50_95": 52.9},
        speed_cpu_ms=85.0,
        description="YOLOv8 Large：高精度检测",
    ),
    "yolov8x.pt": ModelInfo(
        name="yolov8x.pt",
        family="yolov8",
        variant="x",
        task=Task.DETECT,
        backend="ultralytics",
        size_category="xlarge",
        params_m=68.2,
        metrics={"map50_95": 53.9},
        speed_cpu_ms=140.0,
        description="YOLOv8 XLarge：最高精度，计算开销最大",
    ),
}


def get_model_info(name: str) -> ModelInfo | None:
    """按引用名查询模型元数据。

    Args:
        name: 模型引用名，如 ``yolo11n.pt``。

    Returns:
        命中内置或用户扩展目录时返回 ModelInfo，否则返回 None。
    """
    if name in BUILTIN_MODELS:
        return BUILTIN_MODELS[name]
    from od_platform.model_catalog.loader import get_extra_models

    return get_extra_models().get(name)


def list_models(*, task: str | None = None, family: str | None = None) -> list[ModelInfo]:
    """列出目录中的模型，可按任务与系列过滤。

    Args:
        task:   仅返回指定任务的模型；None 返回全部任务。
        family: 仅返回指定系列的模型；None 返回全部系列。

    Returns:
        过滤后的 ModelInfo 列表（保持目录定义顺序）。
    """
    from od_platform.model_catalog.loader import get_extra_models

    merged = {**BUILTIN_MODELS, **get_extra_models()}
    models = list(merged.values())
    if task is not None:
        models = [info for info in models if info.task == task]
    if family is not None:
        models = [info for info in models if info.family == family]
    return models


def list_families(*, task: str | None = None) -> list[str]:
    """列出目录中的模型系列名，可按任务过滤。"""
    families: list[str] = []
    for info in list_models(task=task):
        if info.family not in families:
            families.append(info.family)
    return families
