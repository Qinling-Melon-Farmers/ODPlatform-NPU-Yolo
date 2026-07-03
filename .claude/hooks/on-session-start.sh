#!/usr/bin/env bash
# =============================================================================
# on-session-start.sh — Session 启动时扫描待处理协作信号
#
# 角色无关: 统一展示所有待处理信号，用户根据当前终端角色自行判断。
# 如果 session-role 文件存在则额外显示角色提示。
# =============================================================================
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"

HAS_SIGNAL=0

echo "" >&2
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
echo "  ODPlatform 工作流 — Session 启动检查" >&2
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2

# ── 角色提示 (如果存在 .claude/session-role) ──────────────────
ROLE_FILE=".claude/session-role"
if [ -f "$ROLE_FILE" ]; then
    ROLE=$(cat "$ROLE_FILE")
    case "$ROLE" in
        reviewer)   ICON="🔍" ;;
        developer)  ICON="🔧" ;;
        documenter) ICON="📝" ;;
        *)          ICON="❓" ;;
    esac
    echo "  当前角色: ${ICON} ${ROLE}" >&2
else
    echo "  提示: 创建 .claude/session-role 文件标记当前终端角色" >&2
fi
echo "" >&2

# ── 1. 待 Review 的提交 ───────────────────────────────────────
REVIEW_DIR=".claude/workflow/review-request"
if [ -d "$REVIEW_DIR" ]; then
    PENDING=$(find "$REVIEW_DIR" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l)
    if [ "$PENDING" -gt 0 ]; then
        HAS_SIGNAL=1
        echo "  📬 待审查提交: ${PENDING} 个" >&2
        for f in $(ls -1t "$REVIEW_DIR"/*.json 2>/dev/null); do
            python -c "
import json
with open('$f') as fh:
    d = json.load(fh)
print(f'     └─ {d.get(\"last_commit\",\"?\")}  [{d.get(\"branch\",\"?\")}]  status={d.get(\"status\",\"?\")}')
" 2>/dev/null || true
        done
        echo "     💡 Reviewer: 执行 /review-pending 开始审查" >&2
        echo "" >&2
    fi
fi

# ── 2. 待处理的 Review 反馈 ────────────────────────────────────
DONE_DIR=".claude/workflow/review-done"
if [ -d "$DONE_DIR" ]; then
    PENDING=$(find "$DONE_DIR" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l)
    if [ "$PENDING" -gt 0 ]; then
        HAS_SIGNAL=1
        echo "  📬 Review 反馈: ${PENDING} 个" >&2
        for f in $(ls -1t "$DONE_DIR"/*.json 2>/dev/null); do
            python -c "
import json
with open('$f') as fh:
    d = json.load(fh)
print(f'     └─ {d.get(\"verdict\",\"?\")} — {d.get(\"summary\",\"?\")}')
" 2>/dev/null || true
        done
        echo "     💡 Developer: 根据反馈修改代码后重新提交" >&2
        echo "" >&2
    fi
fi

# ── 3. 待更新的文档 ───────────────────────────────────────────
DOC_DIR=".claude/workflow/doc-request"
if [ -d "$DOC_DIR" ]; then
    PENDING=$(find "$DOC_DIR" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l)
    if [ "$PENDING" -gt 0 ]; then
        HAS_SIGNAL=1
        echo "  📝 待更新文档: ${PENDING} 个事件" >&2
        for f in $(ls -1t "$DOC_DIR"/*.json 2>/dev/null); do
            python -c "
import json
with open('$f') as fh:
    d = json.load(fh)
print(f'     └─ {d.get(\"event\",\"?\")}: {d.get(\"description\",\"?\")}  [{d.get(\"last_commit\",\"?\")}]')
" 2>/dev/null || true
        done
        echo "     💡 Documenter: 基于事件更新实习日志" >&2
        echo "" >&2
    fi
fi

# ── 无信号 ─────────────────────────────────────────────────────
if [ "$HAS_SIGNAL" -eq 0 ]; then
    echo "  ✅ 无待处理信号，工作流状态干净" >&2
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >&2
exit 0
