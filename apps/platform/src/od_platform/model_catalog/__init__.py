"""模型目录子系统：内置模型元数据、用户扩展与推荐。

面向多任务 (detect/classify/segment) 与多后端 (ultralytics/torchvision)
设计，未来引入分类、分割等更多 CV 模型时只需补充目录条目与主指标映射。
"""

from __future__ import annotations

from od_platform.model_catalog.catalog import (
    BUILTIN_MODELS,
    PRIMARY_METRIC,
    ModelInfo,
    get_model_info,
    list_families,
    list_models,
)
from od_platform.model_catalog.loader import get_extra_models
from od_platform.model_catalog.recommender import recommend_model

__all__ = [
    "BUILTIN_MODELS",
    "PRIMARY_METRIC",
    "ModelInfo",
    "get_extra_models",
    "get_model_info",
    "list_families",
    "list_models",
    "recommend_model",
]
