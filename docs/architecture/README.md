# 架构决策记录

本目录存放 ODPlatform 的 ADR（Architecture Decision Records），用于记录关键架构选择和原因。

## 已有 ADR

| 编号 | 标题 | 状态 |
| --- | --- | --- |
| [ADR-001](ADR-001-monorepo.md) | 采用 Monorepo + apps/ 多端布局 | 已采纳 |
| [ADR-002](ADR-002-reset-safety-design.md) | reset_project 安全重置设计 | 已采纳 |
| [ADR-003](ADR-003-data-converter-registry.md) | 数据转换器注册表设计 | 已采纳 |

## 新增规则

新增 ADR 时使用 `ADR-<编号>-<简短描述>.md` 命名，并至少说明：

- 背景和问题。
- 可选方案。
- 当前决策。
- 影响和后续约束。

架构文档应描述长期决策，不记录临时调试步骤。
