#!/usr/bin/env bash
# =============================================================================
# pre-commit-guard.sh — git commit 前置门禁
#
# 触发条件: Bash 工具调用中包含 "git commit"
# 门禁流程:
#   1. ruff check  (代码风格)
#   2. pytest       (单元测试)
#   3. 任一失败 → exit 2 (阻断提交), 全部通过 → exit 0 (放行)
#
# 注意: 只拦截 git commit, 不拦截 git add / git push 等操作.
# =============================================================================
set -euo pipefail

# ── 解析工具输入 ──────────────────────────────────────────────
INPUT=$(cat - 2>/dev/null || echo "{}")
COMMAND=$(echo "$INPUT" | python -c "import sys,json; print(json.load(sys.stdin).get('command',''))" 2>/dev/null || echo "")

# 不是 git commit 则直接放行
if [[ ! "$COMMAND" =~ git[[:space:]]+commit ]]; then
    exit 0
fi

echo "╔══════════════════════════════════════════════════════════╗" >&2
echo "║  pre-commit 门禁启动 …                                  ║" >&2
echo "╚══════════════════════════════════════════════════════════╝" >&2

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/c/Users/ASUS/Desktop/生产实习/ODplatform}"
cd "$PROJECT_DIR"
FAIL=0

# ── Step 1: ruff check ───────────────────────────────────────
echo "" >&2
echo "▸ Step 1/2: ruff check (代码风格) …" >&2

# 找到平台包源码目录
PKG_DIR="apps/platform/src/od_platform"
if [ -d "$PKG_DIR" ]; then
    if command -v ruff &>/dev/null; then
        ruff check "$PKG_DIR" >&2 2>&1 || {
            echo "  ❌ ruff check 未通过 — 请修复后重新提交" >&2
            FAIL=1
        }
    else
        echo "  ⚠️  ruff 未安装, 跳过" >&2
    fi
else
    echo "  ⚠️  未找到源码目录, 跳过 ruff" >&2
fi

# ── Step 2: pytest ───────────────────────────────────────────
echo "" >&2
echo "▸ Step 2/2: pytest (单元测试) …" >&2

if [ -d "apps/platform/tests" ]; then
    # 使用 conda 环境
    if command -v conda &>/dev/null; then
        # conda activate 在非交互式脚本中需要用 source
        CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            conda activate odplat 2>/dev/null || true
        fi
    fi

    python -m pytest apps/platform/tests/ -q --tb=short >&2 2>&1 || {
        echo "  ❌ pytest 未通过 — 请修复后重新提交" >&2
        FAIL=1
    }
else
    echo "  ⚠️  未找到测试目录, 跳过 pytest" >&2
fi

# ── 判定 ─────────────────────────────────────────────────────
echo "" >&2
if [ "$FAIL" -eq 0 ]; then
    echo "✅ 门禁全部通过，允许提交" >&2
    exit 0
else
    echo "╔══════════════════════════════════════════════════════════╗" >&2
    echo "║  ❌ 门禁失败 — git commit 已被阻断                      ║" >&2
    echo "║  请修复以上问题后重新提交                                ║" >&2
    echo "╚══════════════════════════════════════════════════════════╝" >&2
    exit 2
fi
