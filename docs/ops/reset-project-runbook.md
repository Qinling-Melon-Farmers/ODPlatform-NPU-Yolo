# reset_project 运维手册

## 适用范围

`odp-reset` 用于清理 ODPlatform 运行时产物，恢复到接近 `git clone` 后的状态。它不会清理 `data/raw/`、`models/pretrained/`、代码、文档、脚本、`.git/` 和 `apps/platform/meta_logging/`。

## 前置条件

1. 位于 `ODPlatform` 仓库根目录。
2. 已激活 `odplat` 环境。
3. 已完成可编辑安装：`pip install -e .\apps\platform`。

Windows PowerShell 示例：

```powershell
conda activate odplat
python -c "import sys; print(sys.executable)"
pip install -e .\apps\platform
```

确认 Python 路径包含 `envs\odplat`。

## 场景 1：查看将要清理什么

```powershell
odp-reset
```

预期：

- 输出 `[DRY-RUN]` 删除计划。
- 不删除任何文件。
- 末尾提示需要 `--yes` 才会真实删除。

## 场景 2：本地交互式清理

```powershell
odp-reset --yes
```

按提示输入精确大写：

```text
RESET
```

输入其他内容会取消，退出码仍为 0，文件不会被删除。

## 场景 3：CI 无交互清理

```powershell
odp-reset --yes --force
```

退出码含义：

| 退出码 | 含义 |
| --- | --- |
| 0 | dry-run 完成、用户取消，或真实清理全部成功 |
| 1 | 真实清理部分失败 |
| 2 | 全部失败、保护机制拦截或参数错误 |

## 审计日志

审计日志写入：

```text
apps/platform/meta_logging/reset_project/
```

文件名形如：

```text
reset-project_audit_YYYYMMDD-HHMMSS-ffffff_<pid>.log
```

首行 `[AUDIT]` 记录执行者、主机、PID、命令参数、Git 提交、Git 状态、Python 路径等上下文；末行 `[RESULT]` 记录退出码与结果。

## 故障处理

- 如果提示触发 `PROTECTED_DIRS` 校验，立即停止，不要绕过保护；检查 `paths.get_dirs_to_reset()` 是否误加入敏感目录。
- 如果 Windows 上删除失败，先确认文件没有被 IDE、训练进程或终端占用，再重试。
- 如果只想预览，请使用默认命令或显式 `odp-reset --dry-run`。

## 参考

- [ADR-002 reset_project 双层防护与默认安全设计](../architecture/ADR-002-reset-safety-design.md)
