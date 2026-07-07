# Web Backend 端

`apps/web-backend` 是后续 Web 服务端的占位目录。当前项目能力集中在 `apps/platform`，Web 后端尚未正式开发。

## 预期职责

- 对外提供 REST / WebSocket API。
- 调用 `od_platform.inference` 提供在线推理能力。
- 调用 `od_platform.data_validation` 提供数据质检任务接口。
- 管理训练、推理、数据转换任务的队列和状态。

## 当前约束

- 不在 Web 后端重复实现数据转换、质检、训练或推理逻辑。
- 正式开发前只保留 README 和必要占位结构。
- 后续应通过可安装的 `od_platform` 包复用 platform 端能力。

## 参考

- [Platform README](../platform/README.md)
- [ADR-001 Monorepo 决策](../../docs/architecture/ADR-001-monorepo.md)
