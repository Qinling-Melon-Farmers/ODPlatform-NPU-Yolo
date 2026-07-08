"""Text renderers for visualization overlays."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from od_platform.visualization.core.data_types import DrawStyle
from od_platform.visualization.core.text_cache import TextSizeCache, resolve_font_path

logger = logging.getLogger(__name__)


class PillowTextRenderer:
    """Render Chinese/English text on OpenCV BGR images with Pillow."""

    def __init__(self, size_cache: TextSizeCache | None = None) -> None:
        self._size_cache = size_cache
        self._fallback_warned = False

    def set_cache(self, cache: TextSizeCache) -> None:
        """Attach a text size/font cache."""
        self._size_cache = cache

    def render_batch(
        self,
        img: np.ndarray,
        texts: list[tuple[str, tuple[int, int], tuple[int, int, int]]],
        style: DrawStyle,
    ) -> np.ndarray:
        """Render multiple text labels and return a BGR image."""
        if not texts:
            return img

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        font = self._get_font(style)
        for text, position, color_bgr in texts:
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            draw.text(position, text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

    def get_text_size(self, text: str, style: DrawStyle) -> tuple[int, int]:
        """Return text size, preferring the cache when label text is parseable."""
        if self._size_cache is not None:
            parts = text.rsplit(" ", 1)
            if len(parts) == 2:
                return self._size_cache.get_size(parts[0], style.font_size)

        font = self._get_font(style)
        bbox = font.getbbox(text)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

    def _get_font(self, style: DrawStyle) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if self._size_cache is not None:
            return self._size_cache.get_font(style.font_size)

        font_path = resolve_font_path(style.font_path)
        try:
            return ImageFont.truetype(font_path, style.font_size)
        except OSError as exc:
            if not self._fallback_warned:
                logger.warning(
                    "font %r cannot be loaded (%s); fallback PIL default font may not render Chinese correctly",
                    font_path,
                    exc,
                )
                self._fallback_warned = True
            return ImageFont.load_default()
