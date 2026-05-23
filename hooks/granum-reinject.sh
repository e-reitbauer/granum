#!/usr/bin/env bash
# SessionStart (compact) — re-inject top chunks + recent prompt history after compaction
set -euo pipefail

GRANUM_DIR="${GRANUM_CWD:-.}/.granum"
NDJSON="$GRANUM_DIR/chunks.ndjson"
LOG_FILE="$GRANUM_DIR/session.log"

if [[ ! -f "$NDJSON" ]]; then
    exit 0
fi

# Import chunks into ChromaDB and emit top 5 memory + 3 spec chunks
python3 - <<'PYEOF'
import json, os, sys
from pathlib import Path

granum_dir = Path(os.environ.get("GRANUM_CWD", ".")) / ".granum"
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mcp_server.db import GranumDB
    from mcp_server.models import normalize_title
    import hashlib

    config_path = granum_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    project_id = config.get("project_id", "")

    db = GranumDB(
        db_path=granum_dir / "db",
        ndjson_path=granum_dir / "chunks.ndjson",
        stale_threshold_days=config.get("stale_threshold_days", 7),
    )
    db.import_ndjson()

    if not project_id:
        sys.exit(0)

    results = db.query_context(
        project_id=project_id,
        query="project context decisions constraints preferences",
        memory_limit=5,
        spec_limit=3,
        freshness_decay_days=config.get("freshness_decay_days", 90),
    )

    if results:
        print("\n## Granum: Retrieved context")
        for r in results:
            icon = {"decision": "◆", "preference": "★", "file_state": "▪", "constraint": "▲", "spec": "◇"}.get(r["type"], "·")
            stale = " ⚠ STALE" if r.get("stale_warning") else ""
            print(f"{icon} [{r['type']}] {r['title']}{stale}")
            print(f"   {r['content']}")
            if r.get("source"):
                print(f"   Source: {r['source']}")
            print()
except Exception as e:
    print(f"[granum] reinject error: {e}", file=sys.stderr)
PYEOF

# Inject recent prompt history
if [[ -f "$LOG_FILE" ]]; then
    PROMPTS=$(tail -n 10 "$LOG_FILE" 2>/dev/null || true)
    if [[ -n "$PROMPTS" ]]; then
        echo ""
        echo "## Granum: Recent prompt history"
        while IFS= read -r line; do
            echo "- $line"
        done <<< "$PROMPTS"
        echo ""
    fi
fi

exit 0
