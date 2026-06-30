"""系统环境信息采集模块。

采集运行环境的完整信息快照，用于日志记录与实验复现。

@FileName:   system_utils.py
@Function:   环境信息采集 get_basic_device_info / 格式化打印 log_device_info
"""

import logging
import os
import platform
import socket
from datetime import datetime

_logger = logging.getLogger("od_platform.common.system")


def _format_size(bytes_size) -> str:
    """将字节数格式化为人类可读的单位。

    Args:
        bytes_size: 字节数，支持 int / float，为 None 或非数值时返回 ``"N/A"``。

    Returns:
        格式化后的字符串，如 ``"16.0 GB"``、``"512.0 MB"``。
    """
    if bytes_size is None:
        return "N/A"
    try:
        size = float(bytes_size)
    except (TypeError, ValueError):
        return "N/A"
    if size >= 1 << 30:
        return f"{size / (1 << 30):.1f} GB"
    if size >= 1 << 20:
        return f"{size / (1 << 20):.1f} MB"
    if size >= 1 << 10:
        return f"{size / (1 << 10):.1f} KB"
    return f"{size:.0f} B"


def get_basic_device_info() -> dict:
    """采集当前运行环境的结构化信息。

    采集内容按四类组织：

    - **系统信息**：操作系统、主机名、Python 版本、PyTorch 版本、
      Ultralytics 版本、当前时间
    - **CPU 信息**：CPU 型号、逻辑核心数
    - **内存信息**：总内存、可用内存、使用率（psutil 软依赖，缺失时降级）
    - **GPU 信息**：CUDA 可用性、GPU 数量、各 GPU 型号与显存

    Returns:
        包含四个顶层 key 的字典：:

            {
                "系统信息": {...},
                "CPU 信息": {...},
                "内存信息": {...},
                "GPU 信息": {...}
            }

    Note:
        - ``psutil`` 为**软依赖**：未安装时内存字段降级为
          ``"Unknown (psutil 未安装)"``，**不抛异常**。
        - ``torch`` / ``ultralytics`` 缺失时对应字段显示 ``"未安装"``。
        - CPU 核心数使用标准库 ``os.cpu_count()``，不依赖第三方库。
    """
    info: dict = {}

    # ── 系统信息 ──
    system_info: dict = {}
    system_info["操作系统"] = (
        f"{platform.system()} {platform.release()} ({platform.architecture()[0]})"
    )
    system_info["主机名"] = socket.gethostname()
    system_info["Python 版本"] = platform.python_version()

    try:
        import torch
        system_info["PyTorch 版本"] = torch.__version__
    except ImportError:
        system_info["PyTorch 版本"] = "未安装"

    try:
        import ultralytics
        system_info["Ultralytics 版本"] = ultralytics.__version__
    except ImportError:
        system_info["Ultralytics 版本"] = "未安装"

    system_info["当前时间"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info["系统信息"] = system_info

    # ── CPU 信息 ──
    cpu_info: dict = {}
    cpu_info["CPU 型号"] = platform.processor() or "未知"
    cpu_info["逻辑核心数"] = os.cpu_count() or 0
    info["CPU 信息"] = cpu_info

    # ── 内存信息（psutil 为软依赖） ──
    mem_info: dict = {}
    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_info["总内存"] = _format_size(mem.total)
        mem_info["可用内存"] = _format_size(mem.available)
        mem_info["使用率"] = f"{mem.percent:.1f}%"
    except ImportError:
        mem_info["总内存"] = "Unknown (psutil 未安装)"
        mem_info["可用内存"] = "Unknown (psutil 未安装)"
        mem_info["使用率"] = "Unknown (psutil 未安装)"
    info["内存信息"] = mem_info

    # ── GPU 信息 ──
    gpu_info: dict = {}
    try:
        import torch
        gpu_info["CUDA 可用"] = "是" if torch.cuda.is_available() else "否"
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info["GPU 数量"] = gpu_count
            for i in range(gpu_count):
                gpu_info[f"GPU {i} 型号"] = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_mem
                gpu_info[f"GPU {i} 显存"] = _format_size(total_mem)
        else:
            gpu_info["GPU 数量"] = 0
    except ImportError:
        gpu_info["CUDA 可用"] = "未知 (PyTorch 未安装)"
        gpu_info["GPU 数量"] = 0
    info["GPU 信息"] = gpu_info

    return info


def log_device_info(logger: logging.Logger | None = None) -> dict:
    """采集环境信息并以格式化表格输出到日志。

    各类别使用居中标题分隔，中英文键名通过
    :func:`~od_platform.common.string_utils.pad_to_width` 对齐。

    Args:
        logger: 目标 logger 实例，为 None 时使用模块级默认 logger。

    Returns:
        :func:`get_basic_device_info` 返回的原始字典，便于调用方进一步处理。
    """
    from od_platform.common.string_utils import get_display_width, pad_to_width

    log = logger or _logger
    info = get_basic_device_info()
    line_width = 60

    log.info("=" * line_width)
    log.info("环境信息快照".center(line_width))
    log.info("=" * line_width)

    # 统一键名对齐宽度（中文算 2）
    max_key_width = 0
    for category in info.values():
        for key in category:
            w = get_display_width(str(key))
            if w > max_key_width:
                max_key_width = w
    max_key_width += 4

    for category_name, fields in info.items():
        log.info(f"── {category_name} ──".center(line_width, "-"))
        for key, value in fields.items():
            padded_key = pad_to_width(str(key), max_key_width - 2)
            log.info("  %s: %s", padded_key, value)
        log.info("")

    log.info("=" * line_width)
    return info


if __name__ == "__main__":
    # 快速自测：直接运行查看环境信息
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log_device_info()
