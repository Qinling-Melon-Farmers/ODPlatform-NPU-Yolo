# Desktop 端

`apps/desktop` 是后续桌面端应用的占位目录。当前尚未进入正式开发，核心能力先在 `apps/platform` 的 Python 包中实现。

## 预期能力

- 调用 `od_platform.inference` 执行图片、视频和摄像头推理。
- 复用 `od_platform.frame_source` 的统一输入源抽象。
- 展示检测结果、推理速度、帧率和设备信息。
- 提供训练/推理运行配置的可视化编辑入口。

## 当前约束

- 不在 desktop 目录内重复实现训练、推理、数据质检逻辑。
- 桌面端应依赖 `od_platform` 包，而不是复制 platform 代码。
- 正式开发前只保留 README 和必要占位结构。

## 参考

- [Platform README](../platform/README.md)
- [ADR-001 Monorepo 决策](../../docs/architecture/ADR-001-monorepo.md)
