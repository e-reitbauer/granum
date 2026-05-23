#!/usr/bin/env bash
# PostToolUse (Edit|Write) — re-index spec file via MCP server IPC if edited file is a spec
set -euo pipefail

GRANUM_DIR="${GRANUM_CWD:-.}/.granum"
CONFIG_FILE="$GRANUM_DIR/config.json"
EDITED_FILE="${CLAUDE_TOOL_INPUT_PATH:-}"

if [[ -z "$EDITED_FILE" || ! -f "$CONFIG_FILE" ]]; then
    exit 0
fi

python3 - <<PYEOF
import json, os, sys
from pathlib import Path

granum_dir = Path(os.environ.get("GRANUM_CWD", ".")) / ".granum"
cwd = Path(os.environ.get("GRANUM_CWD", "."))
edited_file = Path("${EDITED_FILE}")
sys.path.insert(0, str(granum_dir).replace("/.granum", ""))

try:
    config_path = granum_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    spec_paths = config.get("spec_paths", [])

    if not spec_paths:
        sys.exit(0)

    # Check if edited file is under any spec path
    is_spec = False
    try:
        rel = edited_file.relative_to(cwd)
        for sp in spec_paths:
            if str(rel).startswith(sp) or str(rel) == sp:
                is_spec = True
                break
    except ValueError:
        pass

    if not is_spec:
        sys.exit(0)

    from mcp_server.ipc import ipc_call
    rel_path = str(edited_file.relative_to(cwd))
    result = ipc_call(granum_dir, "reindex_spec_file", {
        "file_path": str(edited_file),
        "rel_path": rel_path,
    })

    if result and result.get("indexed"):
        print(f"\nSpec re-indexed: {rel_path} ({result['indexed']} chunk(s))")
        # Query for potentially conflicting memory chunks
        overlap = ipc_call(granum_dir, "query_context", {
            "query": edited_file.read_text(errors="replace")[:500],
            "limit": 5,
        })
        if overlap:
            conflicts = [r for r in overlap if r.get("final_score", 0) >= 0.6 and r.get("type") != "spec"]
            if conflicts:
                print("Review these memory chunks for conflicts:")
                for r in conflicts:
                    print(f"  {r['id'][:8]}  [{r['type']}] {r['title']}  (score: {r['final_score']})")
        print()

except Exception as e:
    print(f"[granum] spec-sync error: {e}", file=sys.stderr)
PYEOF

exit 0
