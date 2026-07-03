请执行以下审查流程：

## 1. 扫描待审查请求

检查 `.claude/workflow/review-request/` 目录下所有 `.json` 文件，按时间排序。

对每个 status=pending 的请求：
- 读取 `last_commit` 字段获取 commit hash
- 执行 `git show <hash>` 查看变更
- 结合 git diff 做代码审查

## 2. 逐 commit 审查

对每个待审查的 commit，按以下维度评审：

### 正确性 (Critical)
- 逻辑是否正确？边界条件是否覆盖？
- 异常处理是否完整？（不裸写 except:）
- 是否有潜在的 None 引用 / 类型错误？

### 设计 (Important)
- 是否遵循 AGENTS.md 中的架构规则？
- 是否重复代码？是否可以复用已有设施？
- 新模块是否符合 Monorepo + src/ 布局？

### 风格 (Nice-to-have)
- 命名是否符合项目惯例？
- 注释是否充分（中文注释 + 英文标识符）？
- 导入顺序是否规范？

## 3. 输出审查报告

对每个 commit 生成结构化审查意见：

```json
{
  "commit": "<hash>",
  "message": "<commit message>",
  "verdict": "approved|changes_requested|blocked",
  "findings": [
    {"severity": "error|warning|info", "file": "...", "line": N, "description": "..."}
  ],
  "summary": "一句话总结"
}
```

## 4. 写入审查结果

将审查结果写入 `.claude/workflow/review-done/<timestamp>.json`：

```json
{
  "timestamp": "<current utc>",
  "reviewer": "13384",
  "reviewed_commits": [...],
  "verdict": "approved|changes_requested|blocked",
  "findings": [...],
  "summary": "...",
  "status": "done"
}
```

然后把对应的 `review-request` 文件 status 改为 `done` 或删除。

## 5. 统计汇总

审查完成后输出：
- 共审查了 X 个 commit
- approved: A / changes_requested: B / blocked: C
- 关键问题清单（ERROR 级别）
