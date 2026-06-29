# Desktop 端（占位）

| 属性 | 值 |
|------|-----|
| **状态** | Placeholder（占位，未启动） |
| **所属** | ODPlatform · apps/desktop |

---

## 用途

未来将承载 ODPlatform 的桌面客户端（基于 PyQt），提供：

- 实时检测画面显示（调用推理引擎）
- 图片 / 视频文件的本地推理
- 可视化配置界面
- 检测结果的交互式浏览

---

## 依赖

- `od_platform`（通过 `pip install -e ../platform` 在同一环境中复用）

---

## 未来计划

V1.1+ 启动实际开发。当前仅预留目录位置，保证 Monorepo 架构的完整性。

---

## 参考

- [ADR-001 Monorepo 决策](../../../docs/architecture/ADR-001-monorepo.md)
- [platform 端 README](../platform/README.md)
