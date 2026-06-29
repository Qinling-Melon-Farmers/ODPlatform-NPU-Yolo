"""ODPlatform 通用基础设施层。

本层包含所有业务模块共用的工具，遵守**单向无环依赖**原则：
所有业务模块只向下依赖本层；本层不依赖任何业务模块。

子模块一览：

- :mod:`~od_platform.common.paths`          — 工作区根定位（marker file 模式）与路径常量
- :mod:`~od_platform.common.logging_utils`  — 双端（控制台 + 文件）彩色日志配置
- :mod:`~od_platform.common.string_utils`   — CJK 字符宽度感知的字符串格式化与表格对齐
- :mod:`~od_platform.common.system_utils`   — 环境信息采集（OS / CPU / 内存 / GPU 快照）
- :mod:`~od_platform.common.performance_utils` — 高精度计时装饰器（@time_it）
"""
