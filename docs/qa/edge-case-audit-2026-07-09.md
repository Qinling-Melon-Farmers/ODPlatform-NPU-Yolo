# ODPlatform Edge Case 审查

审查日期：2026-07-09

## 范围

- 主仓库：`ODPlatform/`
- 平台包：`apps/platform/src/od_platform`
- 桌面端：`apps/desktop`
- 本阶段不纳入：Web 端开发、`ODPlatform.backup/`、工作区根目录带教参考脚本和压缩包。

## 当前验证结果

```powershell
conda activate odplat
python -m ruff check apps/platform/src/od_platform apps/desktop scripts
python -m pytest apps/platform/src/od_platform/tests -q
python -m pytest apps/platform/src/od_platform/tests --cov=apps/platform/src/od_platform --cov-report=term-missing -q
python -m compileall apps/desktop
python -m pip install -r requirements.txt --dry-run
```

结果：

- ruff：通过。
- 单测：`137 passed`。
- 覆盖率：总覆盖率 `86%`。
- desktop compileall：通过。
- requirements dry-run：可解析；当前环境缺 `openpyxl`，dry-run 会计划安装 `openpyxl` 和 `et_xmlfile`。
- `python -m pip check`：失败项来自当前 conda 环境中其他项目包的残缺依赖，不作为 ODPlatform 本次验证失败。

## 已收敛的非 Web 缺口

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 环境自检 | 已完成 | 新增 `common.environment`，主要 CLI 启动时非阻断 warning 提示是否未使用 `odplat`。 |
| detect / segment 边界 | 已收紧 | 当前端到端只支持 `detect`；`segment` 明确作为未来扩展。 |
| requirements | 已完成 | 根目录 `requirements.txt` 由 `pipreqs` 候选、platform pyproject 和 desktop requirements 合并生成。 |
| reset 备份 | 已完成 | `odp-reset --backup` 会在真实删除前备份到 `runs/reset_backup/<timestamp>/`，dry-run 不创建备份。 |
| 数据指纹 | 已完成 | `write_dataset_yaml()` 在 `odp_meta.fingerprint` 写入样本数、类别数、split 计数和 SHA256 摘要。 |
| YAML schema 测试 | 已补齐 | 覆盖 `nc/names` 不一致、空 names、重复类别名、非法类型。 |
| runtime pipeline config 测试 | 已补齐 | 覆盖缺失配置回退和非法颜色映射。 |
| 视频 stride | 已补齐 | `odp-infer --vid-stride` 已暴露到 CLI，测试确认 threaded video 会传到 frame source。 |
| desktop worker 测试 | 已补齐 | 覆盖后台 CLI 命令构造、PYTHONPATH 注入和停止任务。 |

## 仍保留为未来扩展

| 项目 | 原因 |
| --- | --- |
| Web 端 | 本阶段明确不推进 Web。 |
| `segment` 端到端链路 | 当前数据转换、训练、评估、可视化没有完整闭环。 |
| frame source 注册表 | 当前 factory 已可用，注册表和自动发现留给更多输入源阶段。 |
| 深度相机 / 红外相机 | 需要真实硬件和驱动，不适合当前无设备单测。 |
| 推理 macOS GUI 主线程适配 | 当前主要运行环境是 Windows PowerShell / PyCharm Terminal。 |
| Word 报告 | 已有 JSON / Markdown / HTML / CSV/Excel 返工清单，Word 暂列扩展。 |
| 训练服务大重构 | 当前训练链路可用且有审计/归档；不在收尾阶段做大范围改造。 |
| 桌面端 exe 打包 | 用户已明确先不考虑打包。 |

## 重点风险

- 当前环境里存在多个其他项目包的残缺依赖，`pip check` 会失败；这不是 ODPlatform requirements 的解析问题，但会污染环境健康判断。
- 部分早期模块中文注释在终端可能出现乱码，核心逻辑已通过测试；后续可独立做编码清理，不建议混入功能提交。
- 摄像头、RTSP/HTTP 流、OpenCV 窗口暂停等真实设备/GUI 分支仍以人工 smoke 为主，自动化证据少于纯后端逻辑。

## 建议的下一步

1. 提交本次非 Web 收尾改动。
2. 用 `odp-reset --dry-run --backup` 和 `odp-reset --yes --backup --force` 在临时目录或受控环境再做一次人工验证。
3. 安装 `openpyxl` 后验证数据质检 Excel 返工清单导出。
4. 继续推进桌面端体验优化，但不再扩展底层业务链路。
