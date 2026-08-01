"""OpenCV 交互式标注画布。

鼠标左键拖拽绘制边界框，键盘选择类别与导航，HUD 通过 PIL 渲染
（复用项目字体解析，支持中文类别名）。坐标转换函数为模块级纯函数，
便于无 GUI 环境下单测。

交互约定:
    - 鼠标左键: 按下 -> 拖动 -> 释放，完成一个框
    - 0-9:      选择类别
    - d:        删除当前图最后一个框
    - s/Enter:  保存并进入下一张
    - Space:    跳过当前图（不保存）
    - q/Esc:    退出整个标注会话

@FileName:   canvas.py
@Function:   标注画布、交互循环与坐标缩放
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from od_platform.annotation.writer import BBox, bbox_from_pixels
from od_platform.visualization.core.text_cache import resolve_font_path

logger = logging.getLogger(__name__)

#: 类别调色板（BGR），超出长度时循环取用。
PALETTE: tuple[tuple[int, int, int], ...] = (
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 88, 50),
)


@dataclass(frozen=True)
class CanvasResult:
    """一次画布交互的返回结果。

    Attributes:
        action: 结束方式，``save`` / ``skip`` / ``quit`` 之一。
        boxes:  归一化标注列表；``skip`` 时为空。
    """

    action: str
    boxes: list[BBox] = field(default_factory=list)

    @property
    def is_quit(self) -> bool:
        return self.action == "quit"


def compute_display_scale(image_width: int, image_height: int, max_display_size: int) -> float:
    """计算显示缩放因子，保证显示图不超过 max_display_size 的任一边。

    Args:
        image_width:  原图宽。
        image_height: 原图高。
        max_display_size: 显示尺寸上限（像素）。

    Returns:
        缩放因子，小图返回 1.0（不放大）。
    """
    if image_width <= 0 or image_height <= 0:
        return 1.0
    return min(1.0, max_display_size / image_width, max_display_size / image_height)


class AnnotationCanvas:
    """基于 OpenCV 窗口的标注画布。

    Args:
        classes:           类别名称列表，index 即类别 ID。
        window_name:       OpenCV 窗口标题。
        max_display_size:  显示尺寸上限，超出时等比缩小。
        min_box_size_px:   忽略小于该像素宽高的误触框。
    """

    def __init__(
        self,
        classes: list[str],
        *,
        window_name: str = "odp-annotate",
        max_display_size: int = 1200,
        min_box_size_px: int = 3,
    ) -> None:
        self.classes = list(classes)
        self.window_name = window_name
        self.max_display_size = max_display_size
        self.min_box_size_px = min_box_size_px

        self._scale: float = 1.0
        self._original_size: tuple[int, int] = (0, 0)
        #: 每框为 (x1, y1, x2, y2, class_id)，显示像素坐标。
        self._boxes: list[tuple[int, int, int, int, int]] = []
        self._current_class: int = 0
        self._drag_start: tuple[int, int] | None = None
        self._drag_end: tuple[int, int] | None = None

    # ---- 对外接口 ----

    def annotate_image(self, image_path: Path, existing: list[BBox] | None = None) -> CanvasResult:
        """打开一张图片并交互标注。

        Args:
            image_path: 图片路径。
            existing:   已有标注（编辑模式），按归一化坐标传入。

        Returns:
            CanvasResult：save 带标注、skip 空标注、quit 终止会话。
        """
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("无法读取图片，跳过: %s", image_path)
            return CanvasResult(action="skip")

        height, width = image.shape[:2]
        self._original_size = (width, height)
        self._scale = compute_display_scale(width, height, self.max_display_size)
        display = cv2.resize(image, (0, 0), fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)

        self._boxes = [(*box.to_pixels(width, height), box.class_id) for box in (existing or [])]
        self._drag_start = None
        self._drag_end = None
        self._current_class = 0

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        result: CanvasResult | None = None
        try:
            while result is None:
                frame = self._render(display, image_path.name)
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(20) & 0xFF
                result = self._on_key(key)
        finally:
            cv2.destroyWindow(self.window_name)

        if result.action == "skip":
            return CanvasResult(action="skip")
        return result

    def close(self) -> None:
        """销毁窗口（保险调用，正常退出已自动销毁）。"""
        cv2.destroyWindow(self.window_name)

    # ---- 事件处理 ----

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y)
            self._drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start is not None:
            self._drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drag_start is not None:
            self._drag_end = (x, y)
            self._commit_dragged_box()
            self._drag_start = None
            self._drag_end = None

    def _commit_dragged_box(self) -> None:
        if self._drag_start is None or self._drag_end is None:
            return
        x1, y1 = self._drag_start
        x2, y2 = self._drag_end
        if abs(x2 - x1) < self.min_box_size_px or abs(y2 - y1) < self.min_box_size_px:
            logger.debug("忽略过小拖拽框 (%d, %d)-(%d, %d)", x1, y1, x2, y2)
            return
        self._boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), self._current_class))
        logger.debug("完成框 #%d: 类别 %d, (%d, %d)-(%d, %d)", len(self._boxes), self._current_class, x1, y1, x2, y2)

    def _on_key(self, key: int) -> CanvasResult | None:
        """处理键盘事件，返回 None 表示继续。"""
        if key in (ord("q"), 27):  # q / Esc
            return CanvasResult(action="quit")
        if key in (ord("s"), 13, 10):  # s / Enter
            return CanvasResult(action="save", boxes=self._to_normalized_boxes())
        if key == ord(" "):
            return CanvasResult(action="skip")
        if key in (ord("d"), ord("D")):
            if self._boxes:
                self._boxes.pop()
                logger.debug("删除最后一个框，剩余 %d 个", len(self._boxes))
            return None
        if ord("0") <= key <= ord("9"):
            class_id = key - ord("0")
            if class_id < len(self.classes):
                self._current_class = class_id
                logger.debug("切换类别 -> %d (%s)", class_id, self.classes[class_id])
            return None
        return None

    # ---- 渲染 ----

    def _render(self, display: np.ndarray, image_name: str) -> np.ndarray:
        frame = display.copy()

        for x1, y1, x2, y2, class_id in self._boxes:
            color = PALETTE[class_id % len(PALETTE)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if self._drag_start is not None and self._drag_end is not None:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_end
            cv2.rectangle(frame, (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2)), PALETTE[self._current_class % len(PALETTE)], 1)

        hud_lines = [
            f"image: {image_name}",
            f"class[{self._current_class}]: {self.classes[self._current_class]}",
            f"boxes: {len(self._boxes)}",
            "keys: 0-9 class | d del | s save | space skip | q quit",
        ]
        return _blit_hud(frame, hud_lines)

    def _to_normalized_boxes(self) -> list[BBox]:
        width, height = self._original_size
        boxes: list[BBox] = []
        for x1, y1, x2, y2, class_id in self._boxes:
            boxes.append(
                bbox_from_pixels(
                    class_id,
                    x1 / self._scale,
                    y1 / self._scale,
                    x2 / self._scale,
                    y2 / self._scale,
                    width,
                    height,
                )
            )
        return boxes


@lru_cache(maxsize=8)
def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载字体（缓存），找不到字体时回退 PIL 默认字体。"""
    try:
        return ImageFont.truetype(resolve_font_path(None), font_size)
    except OSError as exc:
        logger.warning("字体加载失败，使用默认字体: %s", exc)
        return ImageFont.load_default()


def _blit_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """在画面左上角叠加中文 HUD 文本。"""
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_font(20)
    y = 8
    for line in lines:
        draw.text((8, y), line, font=font, fill=(0, 255, 0))
        y += 26
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
