#!/usr/bin/env bash
# UserPromptSubmit — log prompt + remind Claude of per-turn granum actions
set -euo pipefail

GRANUM_DIR="${GRANUM_CWD:-.}/.granum"
LOG_FILE="$GRANUM_DIR/session.log"

mkdir -p "$GRANUM_DIR"

PROMPT=$(jq -r '.prompt // empty' 2>/dev/null)
if [[ -z "$PROMPT" ]]; then
    exit 0
fi

echo "$PROMPT" >> "$LOG_FILE"

# Keep last 20 entries
if [[ -f "$LOG_FILE" ]]; then
    tail -n 20 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

CONFIG_FILE="$GRANUM_DIR/config.json"
if [[ ! -f "$CONFIG_FILE" ]]; then
    exit 0
fi

# Auto-reindex specs before Claude responds
python3 - <<'PYEOF' 2>/dev/null
import os, sys
from pathlib import Path
granum_dir = Path(os.environ.get("GRANUM_CWD", ".")).resolve() / ".granum"
sys.path.insert(0, str(granum_dir.parent))
try:
    from mcp_server.ipc import ipc_call
    ipc_call(granum_dir, "reindex_specs", {})
except Exception:
    pass
PYEOF

cat <<'MSG'
[granum] Call query_context with the user's message before acting.

If the user picked an option you presented: call save_context(type=decision|preference) before responding.
If you chose an approach and acted on it: call save_context(type=decision).
If the user corrected you: call save_context(type=decision|constraint, importance=4+) immediately.
If a feature is done or a bug is fixed: call save_context with what happened and why it matters.
After every save_context: call add_edge with the returned ID to link related chunks.
MSG

# Detect session-end phrasing → remind about save_handoff
PROMPT_LOWER=$(echo "$PROMPT" | tr '[:upper:]' '[:lower:]')
if echo "$PROMPT_LOWER" | grep -qE '\b(done|bye|goodbye|thanks|thank you|that.s all|all done|finish|we.re done|good night|see you)\b'; then
    echo "[granum] Looks like end of session — call save_handoff with a 2-3 sentence summary: key decisions, files changed, open questions."
fi

exit 0
