"""性能计时工具模块。

提供通用计时装饰器，自动选择合适的时间单位输出耗时信息。

@FileName:   performance_utils.py
@Function:   性能计时装饰器 time_it
"""

import functools
import logging
import time
from typing import Any, Callable, Optional

_logger = logging.getLogger("od_platform.common.performance")


def _format_duration(seconds: float) -> str:
    """将秒数格式化为人类可读的时间字符串。

    自动选择最合适的单位：
      - < 1 毫秒  → 微秒 (μs)
      - < 1 秒    → 毫秒 (ms)
      - < 60 秒   → 秒 (s)
      - < 1 小时  → 分 + 秒
      - ≥ 1 小时  → 时 + 分 + 秒

    Args:
        seconds: 以秒为单位的时间值。

    Returns:
        格式化后的时间字符串。

    Examples:
        >>> _format_duration(0.0005)
        '500.00 μs'
        >>> _format_duration(0.125)
        '125.00 ms'
        >>> _format_duration(5.3)
        '5.30 s'
        >>> _format_duration(95.7)
        '1 分 35.70 秒'
        >>> _format_duration(3661.0)
        '1 时 1 分 1.00 秒'
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} μs"
    if seconds < 1.0:
        return f"{seconds * 1_000:.2f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    if seconds < 3600.0:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} 分 {secs:.2f} 秒"
    hours = int(seconds // 3600)
    remaining = seconds % 3600
    minutes = int(remaining // 60)
    secs = remaining % 60
    return f"{hours} 时 {minutes} 分 {secs:.2f} 秒"


def time_it(
    iterations: int = 1,
    name: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> Callable:
    """通用计时装饰器工厂。

    对被装饰函数的执行时间进行高精度计时，并通过 logger 输出。
    使用 ``time.perf_counter()`` 保证单调、高精度计时。

    Args:
        iterations: 执行次数。大于 1 时循环执行多次并输出总耗时与平均耗时。
        name: 显示名称，为 None 时自动取被装饰函数的 ``__name__``。
        logger_instance: 自定义 logger，为 None 时使用模块级默认 logger。

    Returns:
        装饰器函数。

    Note:
        - 使用 ``functools.wraps`` 保留被装饰函数的元信息（__name__、__doc__ 等）。
        - **不修改**被装饰函数的返回值与异常传播行为。
        - 函数抛出异常时不计时（异常向外传播，不输出耗时日志）。

    Usage::

        @time_it(iterations=1, name="模型推理")
        def predict(image):
            ...

        @time_it(iterations=100, name="数据校验")
        def validate(data):
            ...
    """
    logger = logger_instance or _logger

    def decorator(func: Callable) -> Callable:
        func_name = name if name is not None else func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if iterations <= 1:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info("%s 耗时: %s", func_name, _format_duration(elapsed))
            else:
                start_total = time.perf_counter()
                for _ in range(iterations):
                    result = func(*args, **kwargs)
                total_elapsed = time.perf_counter() - start_total
                avg_elapsed = total_elapsed / iterations
                logger.info(
                    "%s 执行 %d 次 | 总耗时: %s | 平均耗时: %s",
                    func_name,
                    iterations,
                    _format_duration(total_elapsed),
                    _format_duration(avg_elapsed),
                )
            return result

        return wrapper

    return decorator
