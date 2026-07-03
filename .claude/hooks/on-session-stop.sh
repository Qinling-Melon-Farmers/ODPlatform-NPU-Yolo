#!/usr/bin/env bash
# =============================================================================
# on-session-stop.sh — Session 结束时记录工作检查点
#
# 角色识别: 读取 .claude/session-role 文件 (单行角色名)
#   - 文件不存在 → 跳过角色信息更新
#   - PID 仅作为参考记录，不参与任何逻辑判断
#
# 写入 .claude/workflow/state.json: 信号计数 + 时间戳 + 角色活跃时间
# =============================================================================
set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"

STATE_FILE=".claude/workflow/state.json"
ROLE_FILE=".claude/session-role"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# 读取当前终端角色
ROLE=""
if [ -f "$ROLE_FILE" ]; then
    ROLE=$(head -1 "$ROLE_FILE" | xargs)  # xargs 去除空白
fi

mkdir -p ".claude/workflow"

python -c "
import json, os

state = {}
if os.path.exists('${STATE_FILE}'):
    with open('${STATE_FILE}', 'r') as f:
        state = json.load(f)

# 更新角色最后活跃时间（基于 session-role 文件，不依赖 PID）
role = '${ROLE}'
if role:
    sessions = state.setdefault('sessions', {})
    if role not in sessions:
        sessions[role] = {'description': '', 'last_active': None}
    sessions[role]['last_active'] = '${TIMESTAMP}'
    # PID 仅作为参考记录
    sessions[role]['_last_pid'] = $$

# 统计待处理信号
signals = {}
for sig_type in ['review-request', 'review-done', 'doc-request']:
    sig_dir = f'.claude/workflow/{sig_type}'
    if os.path.isdir(sig_dir):
        count = len([f for f in os.listdir(sig_dir) if f.endswith('.json')])
        signals[sig_type] = count

state['pending_signals'] = signals
state['updated_at'] = '${TIMESTAMP}'

with open('${STATE_FILE}', 'w', encoding='utf-8') as f:
    json.dump(state, f, ensure_ascii=False, indent=2)
" 2>/dev/null || true

# 无论如何不阻断 Session 停止
exit 0
