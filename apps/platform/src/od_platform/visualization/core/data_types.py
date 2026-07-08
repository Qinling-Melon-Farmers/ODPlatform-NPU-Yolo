"""Runtime data and style configuration for detection visualization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from pydantic import BaseModel, ConfigDict, Field, field_validator


@dataclass
class Detection:
    """One detection result in pixel coordinates."""

    box: tuple[int, int, int, int]
    confidence: float
    label: str
    color: tuple[int, int, int] = (0, 255, 0)


class LabelPosition(Enum):
    """Resolved label position relative to the detection box."""

    ABOVE = auto()
    INSIDE_TOP = auto()
    BELOW = auto()


@dataclass
class LabelLayout:
    """Computed label layout for one detection."""

    box: tuple[int, int, int, int]
    text_pos: tuple[int, int]
    position: LabelPosition
    align_right: bool = False
    label_wider: bool = False


class DrawStyle(BaseModel):
    """Style configuration for drawing boxes and labels."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )

    font_path: str | None = Field(default=None, description="Font path or font file name.")
    font_size: int = Field(default=26, gt=0, le=500, description="Text font size in pixels.")
    line_width: int = Field(default=2, gt=0, le=50, description="Detection box line width.")
    padding_x: int = Field(default=10, ge=0, le=500, description="Horizontal label padding.")
    padding_y: int = Field(default=10, ge=0, le=500, description="Vertical label padding.")
    radius: int = Field(default=8, ge=0, le=500, description="Rounded corner radius.")
    text_color: tuple[int, int, int] = Field(default=(0, 0, 0), description="Text color in BGR.")

    @field_validator("text_color")
    @classmethod
    def _validate_color(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(value) != 3:
            raise ValueError(f"text_color must have three BGR components, got {value}")
        for channel in value:
            if not isinstance(channel, int) or not (0 <= channel <= 255):
                raise ValueError(f"text_color channels must be 0-255 integers, got {value}")
        return value

    @classmethod
    def from_image_size(
        cls,
        height: int,
        width: int,
        *,
        ref_dim: int = 720,
        base_font_size: int = 26,
        base_line_width: int = 2,
        base_padding_x: int = 10,
        base_padding_y: int = 10,
        base_radius: int = 8,
        font_scale: float = 1.0,
        **overrides: object,
    ) -> DrawStyle:
        """Build a style scaled to image size, with explicit overrides last."""
        scale = min(height, width) / max(ref_dim, 1)
        params: dict[str, object] = {
            "font_size": max(10, int(base_font_size * scale * font_scale)),
            "line_width": max(1, int(base_line_width * scale)),
            "padding_x": max(5, int(base_padding_x * scale)),
            "padding_y": max(5, int(base_padding_y * scale)),
            "radius": max(3, int(base_radius * scale)),
        }
        params.update(overrides)
        return cls(**params)
