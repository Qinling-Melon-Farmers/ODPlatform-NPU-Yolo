# ODPlatform CLI

## odp-init

初始化 ODPlatform 运行时目录：

```powershell
odp-init
```

开发期也可以直接运行：

```powershell
python scripts/init_project.py
```

## odp-reset

安全清理运行时产物，默认 dry-run，不会删除文件：

```powershell
odp-reset
```

真实执行需要显式确认：

```powershell
odp-reset --yes
```

CI 场景可跳过交互确认：

```powershell
odp-reset --yes --force
```

reset 的安全设计见 [ADR-002](../../../../../docs/architecture/ADR-002-reset-safety-design.md)，操作指引见 [reset_project 运维手册](../../../../../docs/ops/reset-project-runbook.md)。
