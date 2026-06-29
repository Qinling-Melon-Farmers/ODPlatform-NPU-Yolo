# ADR-001：采用 Monorepo + apps/ 多端布局

| 属性 | 值 |
|------|-----|
| **状态** | 已采纳 |
| **决策日期** | 2026-06-29 |
| **决策者** | 架构组 |
| **替代方案** | Multi-repo（每端独立仓库） |

---

## 背景

ODPlatform 当前只有一个端（`platform`——训练 + 推理引擎），但未来计划扩展：

- `web-backend`：网页后端，提供 HTTP API 进行在线推理
- `desktop`：桌面客户端（PyQt），面向非命令行用户

在项目初期就需要决定：所有端放在**一个仓库**里，还是**各自独立仓库**？

---

## 备选方案

### 方案 A：Monorepo（单体仓库）

所有端、共享数据、模型权重、运行产物放在同一个仓库中。

```
ODPlatform/
├── apps/
│   ├── platform/      ← 当前唯一
│   ├── web-backend/   ← 未来
│   └── desktop/       ← 未来
├── data/              ← 共享
├── models/            ← 共享
└── ...
```

### 方案 B：Multi-repo（多仓库）

每个端各自一个独立 Git 仓库，共享资产通过外部存储或包发布复用。

---

## 决定

**采用方案 A：Monorepo + apps/ 布局。**

---

## 理由

1. **团队规模小（4–6 人）**：拆分多仓库会带来跨仓库的版本协调成本，在小团队中得不偿失。
2. **共享资产集中管理**：数据集、模型权重只有一份物理存储，避免多仓库间的同步与一致性隐患。
3. **代码复用简单**：端之间通过 `pip install -e ./apps/platform` 即可直接 `import`，无需额外打包发布。
4. **原子化改动**：当一个功能需要同时改 platform 引擎和 web-backend 接口时，一次 commit 全部覆盖，CI 一次验证。
5. **新人上手快**：一个 `git clone` 获得全部代码，不用理解多个仓库之间的依赖关系。

---

## 后果

### 正面

- 降低协作摩擦，所有代码在同一个仓库中
- 共享资产的"单一权威来源"（Single Source of Truth）
- CI/CD 配置只需维护一套

### 负面

- 仓库体积随数据/模型增长而膨胀（通过 `.gitignore` 排除大文件缓解）
- 随着团队扩大（> 20 人），Monorepo 的构建时间、权限控制会成为瓶颈

### 中性

- 要求开发者遵守严格的模块边界（`apps/` 下的端之间禁止循环依赖）

---

## 撤销条件

当满足以下**任意两条**时，应重新评估是否拆分为 Multi-repo：

- 团队规模超过 20 人
- 单个端的 CI 时间超过 30 分钟
- 多个端需要**独立版本发布周期**（如 platform 每月发版、web-backend 每周发版）
- 出现"改 platform 导致 web-backend 崩溃"的事故频率超过可接受水平

---

## 参考资料

- [Monorepo vs Multi-repo: Which is Right for Your Team?](https://semaphoreci.com/blog/what-is-monorepo)
- D0 设计指南 §6.3 "三天后我想加个网页 demo"
- D0 设计指南 附录 A "Monorepo vs Multi-repo 的完整权衡"
