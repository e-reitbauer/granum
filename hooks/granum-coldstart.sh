#!/usr/bin/env bash
# SessionStart (startup) — re-index specs and run coldstart tasks via MCP server IPC
set -euo pipefail

GRANUM_DIR="${GRANUM_CWD:-.}/.granum"

mkdir -p "$GRANUM_DIR"

# Reset session log on new session
> "$GRANUM_DIR/session.log"

# All heavy work goes through IPC so the MCP server's ChromaDB client does the writes.
# If the server isn't up yet, ipc_call returns None and we skip gracefully.
python3 - <<'PYEOF'
import json, os, sys
from pathlib import Path

granum_dir = Path(os.environ.get("GRANUM_CWD", ".")) / ".granum"
sys.path.insert(0, str(granum_dir).replace("/.granum", ""))

try:
    from mcp_server.ipc import ipc_call

    result = ipc_call(granum_dir, "reindex_specs", {})
    if result is None:
        sys.exit(0)  # server not up yet — skip

    tasks = ipc_call(granum_dir, "coldstart_tasks", {})
    if tasks and tasks.get("changed_specs"):
        print("\nSpec files changed since last session:")
        for path in tasks["changed_specs"]:
            print(f"  - {path}")
        print("\nReview and deprecate any conflicting Granum memory chunks.\n")

except Exception as e:
    print(f"[granum] coldstart error: {e}", file=sys.stderr)
PYEOF

exit 0
