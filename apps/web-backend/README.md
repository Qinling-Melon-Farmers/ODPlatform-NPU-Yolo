# Web Backend 端（占位）

| 属性 | 值 |
|------|-----|
| **状态** | Placeholder（占位，未启动） |
| **所属** | ODPlatform · apps/web-backend |

---

## 用途

未来将承载 ODPlatform 的网页后端服务，提供：

- HTTP API 接口（REST / WebSocket）
- 在线推理服务（调用 platform 引擎）
- 用户管理、任务队列等业务逻辑

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
