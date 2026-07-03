#!/usr/bin/env bash
# =============================================================================
# on-after-commit.sh — git commit 后触发协作信号
#
# 由 PostToolUse hook 调用. 当检测到 git commit 执行成功后:
#   1. 写 review-request → 通知 Reviewer (13384) 有新代码待审查
#   2. 写 doc-request    → 通知 Documenter (41432) 更新实习日志
#
# 信号文件格式: .claude/workflow/<type>/<timestamp>.json
# =============================================================================
set -euo pipefail

# ── 解析工具输入 ──────────────────────────────────────────────
INPUT=$(cat - 2>/dev/null || echo "{}")
COMMAND=$(echo "$INPUT" | python -c "import sys,json; print(json.load(sys.stdin).get('command',''))" 2>/dev/null || echo "")

# 不是 git commit 则直接退出
if [[ ! "$COMMAND" =~ git[[:space:]]+commit ]]; then
    exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/c/Users/ASUS/Desktop/生产实习/ODplatform}"
cd "$PROJECT_DIR"

TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
# 获取最近一次提交 (刚刚执行的那个)
LAST_COMMIT=$(git log -1 --format="%h %s" 2>/dev/null || echo "unknown")

echo "" >&2
echo "▸ 提交后协作信号 …" >&2

# ── 1. Review Request ────────────────────────────────────────
REVIEW_FILE=".claude/workflow/review-request/${TIMESTAMP}.json"
mkdir -p ".claude/workflow/review-request"

python -c "
import json, os
payload = {
    'timestamp': '${TIMESTAMP}',
    'requester': 'developer',
    'branch': '${BRANCH}',
    'last_commit': '${LAST_COMMIT}',
    'project_dir': os.getcwd(),
    'status': 'pending'
}
with open('${REVIEW_FILE}', 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(f'  📝 Review 请求已写入: ${REVIEW_FILE}')
" >&2

# ── 2. Doc Request ───────────────────────────────────────────
DOC_FILE=".claude/workflow/doc-request/${TIMESTAMP}.json"
mkdir -p ".claude/workflow/doc-request"

python -c "
import json, os
payload = {
    'timestamp': '${TIMESTAMP}',
    'requester': 'developer',
    'event': 'commit',
    'branch': '${BRANCH}',
    'last_commit': '${LAST_COMMIT}',
    'description': '新提交需要同步更新实习日志',
    'project_dir': os.getcwd(),
    'status': 'pending'
}
with open('${DOC_FILE}', 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(f'  📝 文档请求已写入: ${DOC_FILE}')
" >&2

echo "  ✅ 协作信号已发出 (Review + Doc)" >&2
exit 0
