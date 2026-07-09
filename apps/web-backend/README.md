# Web Backend 端

`apps/web-backend` 是后续 Web 服务端的占位目录。当前项目能力集中在 `apps/platform`，Web 后端尚未正式开发。

## 建设目标

Web 端的核心价值不是重写桌面端，而是把 `od_platform` 的能力服务化，方便多人协作、远程访问和任务留痕。首版目标应聚焦在“任务调度 + 结果浏览 + 在线推理演示”，训练和长耗时任务只做启动、状态查询和产物查看，不在 Web 层实现算法逻辑。

## 预期职责

- 对外提供 REST / WebSocket API。
- 调用 `od_platform.inference` 提供在线推理能力。
- 调用 `od_platform.data_validation` 提供数据质检任务接口。
- 管理训练、推理、数据转换任务的队列和状态。
- 暴露 runs、models、data_validation、evaluation 等产物的检索和下载接口。
- 为前端提供统一的任务状态、日志流和错误信息。

## 当前约束

- 不在 Web 后端重复实现数据转换、质检、训练或推理逻辑。
- 正式开发前只保留 README 和必要占位结构。
- 后续应通过可安装的 `od_platform` 包复用 platform 端能力。
- Web 后端不得直接写死项目外路径，路径解析继续走 `od_platform.common.paths`。
- 长耗时任务必须异步执行，HTTP 请求只负责创建任务和查询状态。

## 推荐技术路线

- 后端框架：FastAPI。
- 后台任务：第一版使用进程内 `ThreadPoolExecutor` / `ProcessPoolExecutor`，后续需要多人并发时再升级 Celery / Redis Queue。
- 实时通信：WebSocket 用于推送任务日志、推理帧状态和进度；REST 用于配置、任务创建和结果查询。
- 前端框架：React + Vite + TypeScript，或在课程要求更轻时使用 Vue 3 + Vite。不要把前端代码放进 `apps/web-backend`，建议后续新增 `apps/web-frontend`。
- 数据存储：首版使用 JSON manifest + runs 目录扫描；需要用户、多任务检索和历史筛选时再引入 SQLite。

## 分阶段计划

### Phase 1: API 骨架

- 新增 FastAPI 应用入口 `apps/web-backend/src/od_platform_web/main.py`。
- 提供健康检查：`GET /health`。
- 提供系统信息：`GET /api/system/info`，复用 platform 的系统/性能工具。
- 提供产物列表：
  - `GET /api/runs`
  - `GET /api/models`
  - `GET /api/datasets`
- 补充最小单测，验证应用可启动、路由返回结构稳定。

### Phase 2: 任务接口

- 统一任务模型：任务 id、类型、状态、启动时间、结束时间、命令参数、日志路径、产物路径、错误摘要。
- 提供任务创建接口：
  - `POST /api/tasks/validate`
  - `POST /api/tasks/evaluate`
  - `POST /api/tasks/infer`
  - `POST /api/tasks/train`
  - `POST /api/tasks/convert`
- 提供任务查询接口：
  - `GET /api/tasks`
  - `GET /api/tasks/{task_id}`
  - `GET /api/tasks/{task_id}/logs`
- 任务执行层可以先复用已有 CLI，也可以直接调用 platform service；优先选择 service，只有兼容历史脚本时才走 CLI。

### Phase 3: 推理服务化

- 提供图片/视频/文件夹离线推理任务。
- 摄像头实时推理在本机 Web 服务内可做演示，但远程部署时不应默认开放本机摄像头。
- WebSocket 推送推理进度、当前 FPS、检测数量和错误事件。
- 结果帧、视频、JSON 产物统一写入 `runs/inference`。

### Phase 4: 前端工作台

- 首屏做成操作台，不做营销页。
- 主要页面：
  - 数据集：导入、转换、质检报告查看。
  - 模型：权重列表、训练记录、评估结果。
  - 推理：上传图片/视频/文件夹，选择模型，显示结果。
  - 任务：任务队列、日志、产物下载。
- 桌面端已有能力可作为交互原型，但 Web 前端不应依赖 PySide6 代码。

### Phase 5: 多用户与部署

- 增加用户身份、操作审计和权限边界。
- 增加 SQLite/PostgreSQL 任务索引。
- 增加 Dockerfile / docker-compose。
- 将大文件、数据集、模型权重接入对象存储或 Git LFS 外部存储。

## 暂不建议立即实现的内容

- 不建议一开始接入 Celery、Redis、PostgreSQL，全套引入会增加调试成本。
- 不建议把训练过程做成完全实时交互，首版只需要任务启动、日志流和产物查看。
- 不建议把桌面端 UI 逻辑迁移到 Web，两个前端共享 platform service，不共享界面代码。
- 不建议直接开放任意路径读写 API，所有路径必须限制在 `ROOT_DIR`、`DATA_DIR`、`MODELS_DIR`、`RUNS_DIR` 等边界内。

## 参考

- [Platform README](../platform/README.md)
- [ADR-001 Monorepo 决策](../../docs/architecture/ADR-001-monorepo.md)
