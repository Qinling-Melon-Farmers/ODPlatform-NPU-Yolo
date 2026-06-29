# Platform 端（训练 + 推理引擎）

| 属性 | 值 |
|------|-----|
| **状态** | Active（活跃开发中） |
| **所属** | ODPlatform · apps/platform |
| **包名** | `od_platform`（可安装 Python 包） |

---

## 用途

Platform 端是 ODPlatform 的**核心端**，承载目标检测平台的全部功能：

- **通用工具层**（`common/`）：路径定位、日志系统、字符串格式化、环境信息采集、性能计时
- **命令行入口**（`cli/`）：`odp-init` 等项目管理命令
- **业务子系统**（后续阶段）：数据流水线、训练、评估、推理

---

## 安装与使用

```bash
# 在仓库根下，以可编辑模式安装
pip install -e ./apps/platform

# 初始化项目运行时目录
odp-init
```

---

## 未来计划

作为 Monorepo 的主端，platform 的 common 层和 CLI 工具将被未来的 `web-backend` 与 `desktop` 端复用。

---

## 参考

- [D0 设计指南](../../../D0-设计指南.md)
- [D1 需求文档（SRS）](../../../D1-项目初始化系统设计-需求文档SRS.md)
