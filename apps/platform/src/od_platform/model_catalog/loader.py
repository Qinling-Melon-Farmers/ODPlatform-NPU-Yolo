"""用户扩展模型目录加载器。

支持通过 ``apps/platform/configs/models.yaml`` 声明自定义模型，
与内置目录合并。文件缺失时优雅降级（不阻断使用）。

YAML 结构示例::

    models:
      - name: resnet18
        family: resnet
        variant: "18"
        task: classify
        backend: torchvision
        size_category: small
        params_m: 11.7
        metrics:
          top1_acc: 69.8
        speed_cpu_ms: 5.0
        description: ResNet18 分类模型

@FileName:   loader.py
@Function:   加载并合并用户扩展模型目录
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from od_platform.common import paths
from od_platform.model_catalog.catalog import ModelInfo

logger = logging.getLogger(__name__)

_EXTRA_MODELS_PATH: Path = paths.CONFIGS_DIR / "models.yaml"
_extra_models_cache: dict[str, ModelInfo] | None = None


def get_extra_models() -> dict[str, ModelInfo]:
    """返回用户扩展模型目录（懒加载并缓存）。

    Returns:
        用户 ``models.yaml`` 中声明的模型字典；文件缺失时为空字典。
    """
    global _extra_models_cache
    if _extra_models_cache is not None:
        return _extra_models_cache

    _extra_models_cache = _load_extra_models(_EXTRA_MODELS_PATH)
    return _extra_models_cache


def _load_extra_models(path: Path) -> dict[str, ModelInfo]:
    """从指定 yaml 文件解析模型列表，解析失败返回空字典。"""
    if not path.exists():
        logger.debug("用户模型扩展文件不存在，跳过: %s", path)
        return {}

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("用户模型扩展文件解析失败，已忽略: %s (%s)", path, exc)
        return {}

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        logger.warning("用户模型扩展文件缺少 models 列表: %s", path)
        return {}

    registry: dict[str, ModelInfo] = {}
    for index, raw in enumerate(models):
        info = _build_model_info(raw)
        if info is None:
            logger.warning("models.yaml 第 %d 条模型定义无效，已跳过", index + 1)
            continue
        registry[info.name] = info
    logger.info("已加载用户扩展模型 %d 个: %s", len(registry), sorted(registry))
    return registry


def _build_model_info(raw: object) -> ModelInfo | None:
    """从原始字典构造 ModelInfo，字段缺失或类型错误时返回 None。"""
    if not isinstance(raw, dict):
        return None

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        return None
    try:
        return ModelInfo(
            name=name,
            family=str(raw.get("family", "custom")),
            variant=str(raw.get("variant", "")),
            task=str(raw.get("task", "detect")),
            backend=str(raw.get("backend", "custom")),
            size_category=str(raw.get("size_category", "")),
            params_m=float(raw.get("params_m", 0.0)),
            metrics=dict(raw.get("metrics") or {}),
            speed_cpu_ms=float(raw.get("speed_cpu_ms", 0.0)),
            description=str(raw.get("description", "")),
        )
    except (TypeError, ValueError) as exc:
        logger.debug("模型 %s 定义无法转换: %s", name, exc)
        return None
