"""标注相关图像工具（公共读取函数）。

``cv2.imread`` 在 Windows 上无法打开含中文的绝对路径（返回 None），
``imread_unicode`` 改用 numpy 字节读取 + ``cv2.imdecode`` 解码，
供画布预览与复核页共用。

@FileName:   image_utils.py
@Function:   兼容非 ASCII 路径的图像读取
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(image_path: Path) -> np.ndarray | None:
    """读取图片，兼容 Windows 非 ASCII（中文）路径。

    Args:
        image_path: 图片路径。

    Returns:
        BGR 图像数组；读取失败返回 None。
    """
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
