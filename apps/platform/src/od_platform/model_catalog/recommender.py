"""模型推荐引擎。

支持中英文自然语言关键词，结合显式条件 (task/family) 过滤，
按主指标排序返回候选模型。既可被 ``odp-list-models --recommend``
使用，也可被 Agent 工具直接调用。

@FileName:   recommender.py
@Function:   关键词解析、候选过滤与排序推荐
"""

from __future__ import annotations

from od_platform.common.constants import Task
from od_platform.model_catalog.catalog import ModelInfo, list_models

#: 速度优先关键词（升序按 CPU 耗时）。
SPEED_KEYWORDS: tuple[str, ...] = ("最快", "快速", "实时", "fastest", "fast", "speed")
#: 精度优先关键词（降序按主指标）。
ACCURACY_KEYWORDS: tuple[str, ...] = ("最准", "准确", "精确", "accuracy", "accurate", "best", "最强")
#: 均衡优先关键词（优先 medium 档）。
BALANCED_KEYWORDS: tuple[str, ...] = ("平衡", "均衡", "适中", "balanced", "balance", "性价比")
#: 轻量优先关键词（升序按参数量）。
LIGHT_KEYWORDS: tuple[str, ...] = ("轻量", "最小", "lightweight", "tiny")
#: 任务关键词 -> 任务名。classify 为未来扩展预留。
TASK_KEYWORDS: dict[str, str] = {
    "分类": "classify",
    "classify": "classify",
    "检测": Task.DETECT,
    "detect": Task.DETECT,
}
#: 系列关键词 -> 系列名。
FAMILY_KEYWORDS: dict[str, str] = {
    "yolo11": "yolo11",
    "v11": "yolo11",
    "yolov8": "yolov8",
    "v8": "yolov8",
    "resnet": "resnet",
}

_SIZE_CATEGORY_RANK: dict[str, int] = {
    "nano": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
    "xlarge": 4,
}


def recommend_model(
    preference: str = "",
    *,
    task: str = Task.DETECT,
    family: str | None = None,
    limit: int = 5,
) -> list[ModelInfo]:
    """根据自然语言偏好推荐模型。

    Args:
        preference: 自然语言描述，如 ``"最快的"``、``"yolov8 最准"``。
        task:       目标任务，默认检测。
        family:     显式系列过滤；同时可从 preference 中解析。
        limit:      返回数量上限。

    Returns:
        按偏好排序的 ModelInfo 列表；无偏好时按速度升序返回。
    """
    pref = preference.strip().lower()
    parsed_family = _parse_family(pref) or family
    parsed_task = _parse_task(pref) or task

    candidates = list_models(task=parsed_task, family=parsed_family)

    mode = _detect_mode(pref)
    ranked = sorted(candidates, key=_sort_key(mode, parsed_task))
    return ranked[:limit]


def _parse_task(pref: str) -> str | None:
    """从关键词中解析任务名，未命中返回 None。"""
    for keyword, task in TASK_KEYWORDS.items():
        if keyword in pref:
            return task
    return None


def _parse_family(pref: str) -> str | None:
    """从关键词中解析模型系列，未命中返回 None。"""
    for keyword, family in FAMILY_KEYWORDS.items():
        if keyword in pref:
            return family
    return None


def _detect_mode(pref: str) -> str:
    """解析排序偏好，返回 speed/accuracy/balanced/light/default 之一。"""
    if any(keyword in pref for keyword in SPEED_KEYWORDS):
        return "speed"
    if any(keyword in pref for keyword in ACCURACY_KEYWORDS):
        return "accuracy"
    if any(keyword in pref for keyword in BALANCED_KEYWORDS):
        return "balanced"
    if any(keyword in pref for keyword in LIGHT_KEYWORDS):
        return "light"
    return "default"


def _sort_key(mode: str, task: str):
    """返回模式对应的排序键函数。"""

    def by_speed(info: ModelInfo) -> tuple[float, float]:
        return info.speed_cpu_ms, -info.primary_metric if info.primary_metric is not None else 0.0

    def by_accuracy(info: ModelInfo) -> tuple[float, float]:
        metric = info.primary_metric if info.primary_metric is not None else -1.0
        return -metric, info.speed_cpu_ms

    def by_balanced(info: ModelInfo) -> tuple[int, float, float]:
        rank = _SIZE_CATEGORY_RANK.get(info.size_category, 99)
        distance = abs(rank - _SIZE_CATEGORY_RANK.get("medium", 2))
        metric = info.primary_metric if info.primary_metric is not None else 0.0
        return distance, -metric, info.speed_cpu_ms

    def by_light(info: ModelInfo) -> tuple[float, float]:
        return info.params_m, info.speed_cpu_ms

    if mode == "speed":
        return by_speed
    if mode == "accuracy":
        return by_accuracy
    if mode == "balanced":
        return by_balanced
    if mode == "light":
        return by_light
    return by_speed
