# ADR-002 reset_project 双层防护与默认安全设计

## 状态

Accepted

## 决策日期

2026-06-30

## 背景

`init_project` 已经能创建运行时目录，但清理仍依赖手工删除。手工 `rm -rf` / `Remove-Item` 容易误删代码、原始数据、预训练权重或 `.git`，且没有审计记录。`reset_project` 是反向工具，目标是把工作区恢复到接近 `git clone` 后状态。

## 备选方案

1. 仅提供 dry-run，不提供删除能力：风险低，但无法满足 CI 和重复实验清理。
2. 仅使用黑名单：能兜住部分危险路径，但删除范围不够明确。
3. 白名单 + 黑名单 + 默认 dry-run：删除范围显式，危险路径二次拦截，默认不破坏。

## 决定

采用方案 3：

- 默认执行 dry-run，只有显式 `--yes` 才进入删除流程。
- `--yes` 默认还需要输入大写 `RESET`，`--yes --force` 用于 CI 无交互场景。
- 删除范围唯一来自 `paths.get_dirs_to_reset()`。
- 删除前逐项调用 `paths.is_protected()`，命中 `PROTECTED_DIRS` 时 fail-fast，任何目录都不删除。
- reset 自身日志写入 `META_LOGGING_DIR`，与会被清理的业务 `LOGGING_DIR` 隔离。
- 清理目录内容时保留顶层 `README.md` / `.gitkeep` 占位文件，保持 Git 工作区接近克隆后的状态。

## 理由

双层防护把“可删什么”和“绝不可删什么”分开维护。即使白名单被错误加入 `.git`、`data/raw` 等敏感路径，黑名单仍会在真实删除前拦截。默认 dry-run 借鉴 `git clean -n`、`terraform plan` 等惯例，降低新人误操作风险。

## 后果

正面：

- 支持本地和 CI 两类场景。
- 审计日志可追踪每次 reset 的操作者、环境和结果。
- 保护原始数据、预训练权重、代码、文档和 Git 元数据。

负面：

- 比直接删除脚本复杂，需要维护白名单和黑名单。
- 真实删除前会扫描目录，大目录场景有少量时间成本。

中性：

- 若需要调整 reset 范围，只修改 `get_dirs_to_reset()`。
- 若需要调整保护范围，只修改 `PROTECTED_DIRS`。

## 撤销条件

只有在平台提供更强的统一清理框架，且能同时满足默认安全、双层防护、审计追踪和跨平台要求时，才考虑替换本设计。

## 参考资料

- `git clean -n`
- `terraform plan`
- `kubectl --dry-run`
- D2 reset_project 需求文档
