"""性能模块烟雾测试 —— 验证 @time_it 装饰器各项功能。"""
# ruff: noqa: E402, I001

import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "platform" / "src"))

from od_platform.common.performance_utils import _format_duration, time_it

SEP = "=" * 60
logging.basicConfig(level=logging.INFO, format="%(message)s")

# 1. 时间格式化边界值
print(SEP)
print("1. _format_duration 五级单位边界测试")
print(SEP)

cases = [
    (0.0005,    "< 1ms  -> us"),
    (0.125,     "< 1s   -> ms"),
    (5.3,       "< 60s  -> s"),
    (95.7,      "< 1h   -> min+sec"),
    (3661.0,    ">= 1h  -> h+min+sec"),
    (0.0,       "边界: 0"),
    (0.000999,  "边界: 0.999ms"),
    (0.001,     "边界: 1ms"),
    (0.999,     "边界: 0.999s"),
    (1.0,       "边界: 1s"),
    (59.999,    "边界: 59.999s"),
    (60.0,      "边界: 60s"),
    (3599.999,  "边界: 3599.999s"),
    (3600.0,    "边界: 3600s"),
]

for secs, desc in cases:
    print(f"  {desc:24s} | {secs:10.4f}s  ->  {_format_duration(secs)}")

# 2. 装饰器烟雾测试
print()
print(SEP)
print("2. @time_it 烟雾测试")
print(SEP)

@time_it(iterations=1, name="单次(sleep 2ms)")
def slow():
    time.sleep(0.002)

@time_it(iterations=100, name="百次平均(sum 1000)")
def fast():
    return sum(range(1000))

assert slow() is None
assert fast() == 499500

# 3. 返回值透传
print()
print(SEP)
print("3. 返回值透传 + 异常不吞")
print(SEP)

@time_it(iterations=1, name="返回字典")
def returns_dict():
    return {"status": "ok", "count": 42}

result = returns_dict()
assert result["status"] == "ok"
assert result["count"] == 42
print("  返回值透传: OK")

@time_it(iterations=1, name="抛出异常")
def raises_error():
    raise ValueError("测试异常——应向外传播")

try:
    raises_error()
    raise AssertionError("应该抛异常但没抛")
except ValueError as e:
    print(f"  异常不吞: OK (捕获到: {e})")

# 4. functools.wraps 元信息
print()
print(SEP)
print("4. functools.wraps 元信息保留")
print(SEP)

@time_it(iterations=1)
def documented_func():
    """这是被装饰函数的文档字符串。"""
    pass

assert documented_func.__name__ == "documented_func"
assert "文档字符串" in (documented_func.__doc__ or "")
print(f"  __name__: {documented_func.__name__}  OK")
print(f"  __doc__ : {documented_func.__doc__}")

# 5. 汇总
print()
print(SEP)
print("全部烟雾测试通过！")
print(SEP)
