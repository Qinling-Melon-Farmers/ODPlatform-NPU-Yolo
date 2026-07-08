"""Pipeline configuration for frame source and visualization settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from od_platform.common import paths
from od_platform.frame_source import CameraConfig

logger = logging.getLogger(__name__)


def _to_bgr_tuple(value: Any, *, field_name: str) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        color = tuple(int(channel) for channel in value)
        if all(0 <= channel <= 255 for channel in color):
            return color
    raise ValueError(f"{field_name} must be a 3-item BGR color, got {value!r}")


@dataclass
class PipelineConfig:
    """Frame-source and visualization configuration loaded from infer_pipeline.yaml."""

    camera_raw: dict[str, Any] = field(default_factory=dict)
    viz_enabled: bool = True
    use_label_mapping: bool = True
    label_mapping: dict[str, str] = field(default_factory=dict)
    color_mapping: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    default_color: tuple[int, int, int] = (0, 255, 0)
    font_path: str | None = None
    style_overrides: dict[str, Any] = field(default_factory=dict)

    def build_camera_config(self) -> CameraConfig | None:
        """Build a CameraConfig when camera settings are provided."""
        if not self.camera_raw:
            return None
        try:
            return CameraConfig(**self.camera_raw)
        except Exception as exc:
            logger.warning("invalid camera config, use frame_source defaults: %s", exc)
            return None

    def normalized_style_overrides(self) -> dict[str, Any]:
        """Return style overrides accepted by DrawStyle.from_image_size."""
        aliases = {
            "box_thickness": "line_width",
            "text_scale": "font_scale",
        }
        allowed = {
            "font_path",
            "font_size",
            "line_width",
            "padding_x",
            "padding_y",
            "radius",
            "text_color",
            "ref_dim",
            "base_font_size",
            "base_line_width",
            "base_padding_x",
            "base_padding_y",
            "base_radius",
            "font_scale",
        }
        normalized: dict[str, Any] = {}
        for key, value in self.style_overrides.items():
            target = aliases.get(key, key)
            if target not in allowed:
                logger.debug("ignore unsupported visualization style option: %s", key)
                continue
            if target == "text_color":
                value = _to_bgr_tuple(value, field_name="text_color")
            normalized[target] = value
        if self.font_path and "font_path" not in normalized:
            normalized["font_path"] = self.font_path
        return normalized

    def to_audit(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot for odp_audit.json."""
        return {
            "viz_enabled": self.viz_enabled,
            "use_label_mapping": self.use_label_mapping,
            "label_mapping_count": len(self.label_mapping),
            "color_mapping_count": len(self.color_mapping),
            "default_color": list(self.default_color),
            "font_path": self.font_path,
            "camera": dict(self.camera_raw),
            "style_overrides": dict(self.style_overrides),
        }


def load_pipeline_config(yaml_path: str | Path | None = None) -> PipelineConfig:
    """Load infer pipeline config. Missing files fall back to defaults."""
    path = paths.runtime_config_path("infer_pipeline") if yaml_path is None else Path(yaml_path)
    if not path.exists():
        logger.warning("pipeline config not found, using defaults: %s", path)
        return PipelineConfig()

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("failed to parse pipeline config, using defaults: %s", exc)
        return PipelineConfig()
    if not isinstance(raw, dict):
        logger.warning("pipeline config top-level must be a mapping, using defaults: %s", path)
        return PipelineConfig()

    frame_source = raw.get("frame_source") or {}
    visualization = raw.get("visualization") or {}
    if not isinstance(frame_source, dict) or not isinstance(visualization, dict):
        logger.warning("pipeline config sections must be mappings, using defaults: %s", path)
        return PipelineConfig()

    colors = visualization.get("color_mapping") or {}
    if not isinstance(colors, dict):
        raise ValueError("visualization.color_mapping must be a mapping")

    default_color = (0, 255, 0)
    if "default_color" in visualization:
        default_color = _to_bgr_tuple(visualization["default_color"], field_name="default_color")

    return PipelineConfig(
        camera_raw=dict(frame_source.get("camera") or {}),
        viz_enabled=bool(visualization.get("enabled", True)),
        use_label_mapping=bool(visualization.get("use_label_mapping", True)),
        label_mapping={str(k): str(v) for k, v in dict(visualization.get("label_mapping") or {}).items()},
        color_mapping={
            str(k): _to_bgr_tuple(v, field_name=f"color_mapping.{k}") for k, v in colors.items()
        },
        default_color=default_color,
        font_path=visualization.get("font_path"),
        style_overrides=dict(visualization.get("style") or {}),
    )
