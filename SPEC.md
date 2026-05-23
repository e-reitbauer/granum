# Granum — Spec

## Overview

Granum is a Claude Code plugin that replaces context-based memory with a persistent semantic vector database. Claude saves important decisions, preferences, and state as chunks during a session. Before each response, relevant chunks are retrieved and injected. After each response, context is compacted when needed. The user sees a normal Claude Code conversation.

---

## Tech Stack

| Component | Technology |
|---|---|
| MCP server | Python, `mcp` SDK |
| Vector store | ChromaDB (local, on-disk) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384d, cosine space) |
| CLI | Python, `typer` + `rich` |
| Hooks | Bash scripts |
| Packaging | Claude Code plugin |

---

## Chunk Schema

```json
{
  "id": "<sha256(project_id + type + normalized_title)>",
  "project_id": "<md5(git_root_path + ':' + git_branch)>",
  "title": "update to auth flow",
  "content": "auth now uses OAuth2, refresh tokens stored in Redis, entry point is backend/auth.py",
  "type": "decision | preference | file_state | constraint | spec",
  "source": null,
  "importance": 3,
  "status": "active | deprecated",
  "created_at": "2026-05-22T10:00:00Z",
  "updated_at": "2026-05-22T14:30:00Z",
  "deleted_at": null
}
```

**Chunk ID:** `sha256(project_id + type + normalized_title)` — deterministic, stable, merge-safe.

**Title normalization:** lowercase, strip all punctuation, collapse whitespace. E.g. `"Auth Flow!"` → `"auth flow"`. Applied consistently everywhere.

**Embedding target:** `type + ": " + normalized_title` — e.g. `"constraint: disable redis caching"`. Embeddings for retrieval only, never identity.

**Project isolation:** `project_id` is `md5(git_root_path + ":" + current_git_branch)`.

**ChromaDB collection:** initialized with `metric="cosine"` explicitly.

**Metadata filtering:** `project_id`, `type`, and `status` filtered at ChromaDB query level.

**Importance:** integer 1–5, Claude-assigned. Default 3.

**Status:** `active` or `deprecated`. Deprecated chunks excluded from retrieval, retained in export.

**Tombstones:** deleted chunks retain `deleted_at` in `.granum/chunks.ndjson`. Never resurrected on import.

**`type: spec`:** read-only chunks sourced from spec files. Written only by the indexer, never by `save_context`. Not exported to `chunks.ndjson` — re-indexed from source files on each startup. `source` field contains the originating file path and section heading.

---

## Configuration — `.granum/config.json`

Created by `granum init`. Committed to git.

```json
{
  "project_id": "abc123_main",
  "spec_paths": ["openspec/specs/", "docs/", "src/specs/"],
  "compaction_threshold": 50,
  "stale_threshold_days": 7,
  "freshness_decay_days": 90,
  "spec_retrieval_limit": 3,
  "memory_retrieval_limit": 7
}
```

All tunables live here. Editable directly or via `granum config set <key> <value>`.

---

## Spec File Detection & Indexing

Granum is aware of external spec files (OpenSpec, custom docs, AGENTS.md, etc.). Spec files are indexed into ChromaDB as `type: spec` chunks — queryable alongside memory chunks but never overwritten by Claude.

### Detection strategy (hybrid)

**Step 1 — known patterns** (checked automatically):
- `openspec/specs/`
- `docs/`
- `AGENTS.md`
- `CLAUDE.md`
- `.cursorrules`
- `GEMINI.md`
- Any top-level markdown containing "SHALL", "MUST", "Given/When/Then"

**Step 2 — `granum init`** confirms detected paths and allows user to add custom ones. Writes final list to `spec_paths` in `.granum/config.json`.

**Step 3 — override** — user can edit `spec_paths` directly or run `granum init --reset` to re-detect.

### Indexing

On `granum init` and `granum-coldstart.sh`:
- Scan all files under `spec_paths`
- Chunk by section/header (files >200 lines)
- Save as `type: spec` chunks with `source: "<file>#<section heading>"`
- Never exported to `chunks.ndjson` — re-indexed from source on every startup
- Existing `type: spec` chunks for the project are cleared and rebuilt on each startup to stay in sync

### Spec change detection

On `SessionStart` (startup matcher), after re-indexing, Granum checks git diff of files under `spec_paths` since `last_session`. If changes found, injects warning:

```
Spec files changed since last session:
  - openspec/specs/auth.md

Review and deprecate any conflicting Granum memory chunks.
```

### PostToolUse hook on spec file writes

When Claude edits a file matching `spec_paths`:

```bash
# granum-spec-sync.sh (PostToolUse, matcher: Edit|Write)
# If edited path matches any entry in spec_paths:
#   → re-index that file's spec chunks immediately
#   → query for memory chunks semantically overlapping the changed section
#   → inject warning: "Spec file re-indexed. Review these memory chunks for conflicts: [list]"
# Claude decides to deprecate/update/keep affected memory chunks
```

---

## Collaborative Sharing

Chunks exported as `.granum/chunks.ndjson` (NDJSON — one chunk per line), committed to git.

- Auto-imported on `SessionStart`
- Auto-exported after every `save_context` or `cleanup_context`
- NDJSON: git diffs are line-level, conflicts are rare and readable
- Branch memory isolated by `project_id` — git handles merging
- Idempotent imports via deterministic IDs
- Tombstones respected on import

---

## MCP Tools

### `query_context(query, type_filter?, limit?)`
- Normalizes and embeds `query`
- Runs two separate queries against ChromaDB, both filtered by `project_id`:
  - **Memory query:** filters `type` in `[decision, preference, file_state, constraint]`, `status: active` — returns top `memory_retrieval_limit` (default 7), scored by `similarity * 0.8 + freshness * 0.2`, weighted by `importance`, per-type quota max 3
  - **Spec query:** filters `type: spec` — returns top `spec_retrieval_limit` (default 3), scored by similarity only (no freshness decay — spec files are maintained externally)
- Merges results, preserving separate quotas so spec chunks never crowd out memory chunks
- Response per chunk:
  ```json
  {
    "id": "...",
    "title": "...",
    "content": "...",
    "type": "...",
    "source": "openspec/specs/auth.md#OAuth Flow",
    "importance": 3,
    "status": "active",
    "age": "3 days",
    "stale_warning": true,
    "similarity": 0.94,
    "final_score": 0.89
  }
  ```
- `stale_warning: true` if `updated_at` older than `stale_threshold_days` (memory chunks only)
- `source` populated for `type: spec` chunks, null for memory chunks

### `save_context(title, content, type, importance?)`
- Rejects `type: spec` — spec chunks are read-only, written only by the indexer
- Normalizes title, computes deterministic ID
- If ID exists: update `content`, `updated_at`, `importance`
- If not: insert with `status: active`
- Embeds and stores vector
- Exports to `.granum/chunks.ndjson`
- Returns: `{ "action": "created | updated", "id": "..." }`

### `cleanup_context(action, chunk_ids, merged_title?, merged_content?, merged_type?, merged_importance?)`
- `delete`: soft-delete (sets `deleted_at`), removes from ChromaDB index
- `deprecate`: sets `status: deprecated`
- `merge`: deprecates all, inserts one new chunk
- `update`: rewrites content of single chunk
- Exports to `.granum/chunks.ndjson`

---

## Hooks

All hooks: timeout protection, fail-open (exit 0 on error), stderr logging.

### `UserPromptSubmit`
```bash
# granum-log.sh
# Appends current prompt to .granum/session.log (keeps last 20 entries)
# Logging only — no querying, no stdout injection
# Claude queries context itself at the start of each turn (see CLAUDE.md)
# Timeout: 2s
```

### `Stop`
```bash
# granum-compact.sh
# Increments .granum/tool_call_count
# If >= compaction_threshold: triggers /compact, resets counter
# Safety net only — Claude self-triggers first
# Timeout: 10s
# Note: replace with CLAUDE_CONTEXT_PERCENT when Anthropic ships it
```

### `SessionStart` (matcher: `compact`)
```bash
# granum-reinject.sh
# Imports .granum/chunks.ndjson
# Re-injects top 5 highest scored chunks via stdout
# Also injects last 10 entries from session.log as "recent prompt history"
# Gives Claude both: what was decided (chunks) + what user was working on (prompts)
# Timeout: 10s
```

Injected format after compaction:
```
## Granum: Recent prompt history
- add error handling to auth middleware
- refactor the Redis connection pool
- write tests for the OAuth flow

## Granum: Retrieved context
[chunks...]
```

### `SessionStart` (matcher: `startup`)
```bash
# granum-coldstart.sh
# Imports .granum/chunks.ndjson
# Checks git diff of spec_paths since last_session → injects change warnings
# If zero active chunks: cold start scan
#   - Reads spec_paths from config first (highest quality source)
#   - Falls back to README, package.json / pyproject.toml, main entry point
#   - Files >200 lines chunked by section/header
#   - Saves architecture summaries, constraints, entrypoint overviews only
# Updates last_session timestamp in config
# Resets .granum/session.log (new session = fresh prompt history)
# Timeout: 10s
```

### `PostToolUse` (matcher: `Edit|Write`)
```bash
# granum-spec-sync.sh
# If edited file path matches any entry in spec_paths:
#   → find semantically overlapping Granum chunks
#   → inject conflict warning into context
# Claude decides to deprecate/update/keep affected chunks
# Timeout: 5s
```

---

## CLAUDE.md Instructions

```
## Granum — Project Memory

Granum is not an optional add-on. It is how you understand this project.
Treat it like a mandatory first step on every turn, and a mandatory last step
after every meaningful action.

---

### TURN START — always do this first

Call query_context with the user's prompt before doing anything else.
Read every returned chunk. They represent decisions already made, constraints
already identified, and context already established. Do not ignore them.
Do not start work without reading them.

If the results feel incomplete for the task at hand, call query_context again
with a more specific term or a higher limit. A second query is cheap.
A wrong assumption is expensive.

Discard chunks with final_score < 0.6 — they are noise.
Treat chunks with stale_warning: true with caution — verify before acting on them.

Spec chunks (type: spec) are sourced from the project's spec files.
Use them for system behavior questions. Use memory chunks for implementation
decisions made during coding. Never modify spec chunks.

---

### TURN END — always do this after meaningful work

After every turn where you made a decision, changed something, learned something,
or identified a constraint — save it.

Save aggressively. When in doubt, save. The cost of an extra chunk is zero.
The cost of losing context across compaction is a broken session.

Save for any of the following:
- You made an architectural or design decision
- You changed how a file, module, or system works
- You learned a user preference or working style
- You identified something that must never happen (constraint)
- You resolved a bug in a non-obvious way
- You chose one approach over another for a reason

Do NOT save:
- Trivial actions: "added a comment", "fixed a typo", "ran a test"
- Things already documented in spec files (type: spec chunks cover those)
- Transient state that won't matter next session

Title format: short, specific, lowercase. "auth flow updated to OAuth2" not "change".
Type: decision | preference | file_state | constraint
Importance: 1–5. Architectural decisions = 5. Minor preferences = 1. Default 3.
Content: max ~400 tokens. Split into multiple chunks if more detail is needed.
Prefer many short focused chunks over few long ones. A chunk about "auth flow"
and a chunk about "Redis session store" are better than one chunk about
"auth and session handling". Short chunks retrieve more precisely.

Same title + type = overwrites the existing chunk. This is intentional.
Use it to update stale information rather than leaving duplicates.

---

### MAINTENANCE — do this when you notice problems

If retrieved chunks contradict each other: deprecate the older one immediately.
If retrieved chunks are stale and wrong: deprecate or update them immediately.
If retrieved chunks are stale but still accurate: call save_context with the same
title and type to refresh the timestamp.
If you see clusters of related low-importance (1-2) stale chunks: delete them.

Use cleanup_context for pruning and merging. Not for undoing decisions.
To change a decision, update the chunk content. Don't delete it.

---

### COMPACTION

Trigger /compact yourself when context feels long or cluttered.
Do not wait for the hook. Proactive compaction keeps sessions clean.
After compaction, Granum re-injects your top chunks and recent prompt history
automatically — you will not lose project context.
```

---

## CLI Style

**Library:** `rich` for all output. No plain print statements anywhere.

**Accent color:** `#da7756` (Claude brand terra cotta orange). Used for:
- Command headers and section titles
- Progress bar fill
- Highlighted values (chunk IDs, important counts)
- Spinner animations

**Color palette:**
| Element | Color |
|---|---|
| Accent / headers | `#da7756` (Claude orange) |
| Success | `#22c55e` (green) |
| Warning / STALE | `#f59e0b` (amber) |
| Error / critical | `#ef4444` (red) |
| Deprecated | `#6b7280` (dim gray) |
| Body text | `#e5e5e5` (off-white) |
| Subtle / metadata | `#9ca3af` (muted gray) |

**Reactivity — every operation shows feedback:**
- Any operation touching the DB shows a progress bar or spinner
- Embedding model load on first run: animated progress bar with download size
- `granum init` spec scan: live file-by-file output as it finds paths
- `granum import`: progress bar per chunk batch
- `granum audit`: animated scan with findings revealed progressively
- `granum timeline`: renders week by week with a brief animation
- All destructive actions (`delete`, `clear`): confirmation prompt with orange warning text

**Spinners:** used for operations with unknown duration (embedding, ChromaDB queries).
Style: `dots` variant from rich, colored `#da7756`.

**Tables:** all tabular output uses `rich.Table` with:
- Header row in Claude orange
- Row striping subtle (no jarring contrast)
- STALE badge: amber if 7–14d, red if >14d
- DEPRECATED rows: dim gray, italicized

**Error messages:** always explain what failed and what to do next. Never just an exception trace.
```
Error: ChromaDB lock detected — is another granum process running?
Run: granum server status
```

**Icons:** Unicode only — no emoji, no Nerd Fonts. Works on every terminal.

| Context | Icon | Unicode |
|---|---|---|
| Success | `✓` | U+2713 |
| Error | `✗` | U+2717 |
| Warning / stale | `⚠` | U+26A0 |
| Info / neutral | `·` | U+00B7 |
| Decision chunk | `◆` | U+25C6 |
| Constraint chunk | `▲` | U+25B2 |
| Preference chunk | `★` | U+2605 |
| File state chunk | `▪` | U+25AA |
| Spec chunk | `◇` | U+25C7 |
| Deprecated | `○` | U+25CB |
| Deleted / tombstone | `×` | U+00D7 |
| Arrow / pointer | `→` | U+2192 |
| Section separator | `─` | U+2500 |

Applied consistently: chunk type always shown as icon + label in tables,
status always shown as icon only in tight views.

---

## CLI — `granum`

```
granum init                              # setup: detect specs, write config
granum init --reset                      # re-run detection
granum config set <key> <value>          # edit config tunables
granum list     [--project <path>] [--type <type>] [--show-deprecated]
granum recent   [--n <count>]            # last N updated chunks, newest first
granum stale    [--project <path>]       # all chunks with stale_warning, sorted by age
granum audit    [--project <path>]       # project memory health report
granum timeline [--project <path>]       # heatmap of chunk activity over time
granum search   <query> [--project <path>]
granum delete   <id>
granum stats    [--project <path>]
granum clear    [--project <path>]
granum export   [--project <path>]
granum import   [--project <path>]
granum server start                      # manually start MCP server (debug)
granum server stop                       # stop manually started server
granum server status                     # show pid, port, running state
granum server logs                       # tail MCP server stderr
```

### `init`
```
$ granum init

Scanning for spec files...
  ✓ Found: openspec/specs/
  ✓ Found: docs/

Are these correct? [Y/n]
Any additional spec paths? (comma-separated, leave blank to skip): src/specs/

Config written to .granum/config.json
```

### `list`
Tabular output. STALE: amber if 7–14d, red if >14d. DEPRECATED: dim gray.
```
  Type         Title                          Age    Imp  Status
◆ decision     auth uses OAuth2               2h     5    ✓
▪ file_state   router.py refactored           1d     3    ✓
★ preference   user prefers async handlers    14d    2    ⚠ STALE
○ decision     old JWT approach               3d     4    deprecated
```

### `search <query>`
Semantic search with `final_score`, `similarity`, `age` columns.

### `delete <id>`
Soft-deletes (sets `deleted_at`). Asks for confirmation.

### `stats`
```
Project:   /home/erik/projects/myapp (branch: main)
Spec paths: openspec/specs/, docs/
Chunks:    42 active, 5 deprecated, 2 deleted
Stale:     3
DB size:   1.2 MB
Embedding: all-MiniLM-L6-v2 (384d)
```

### `recent`
Last N updated chunks, newest first. Default 10.
```
$ granum recent --n 5

  Type         Title                          Updated
◆ decision     auth uses OAuth2               2m ago
▪ file_state   router.py refactored           1h ago
▲ constraint   never write to .env directly   3h ago
★ preference   user prefers async handlers    yesterday
◆ decision     Redis for session store        2d ago
```

### `stale`
All chunks with `stale_warning: true`, sorted oldest first. Actionable list for cleanup.
```
$ granum stale

  Type         Title                          Age    Imp
★ preference   user prefers async handlers    14d    2
▪ file_state   old database schema            21d    3
◆ decision     initial Redis config           30d    4
```

### `timeline`
GitHub-style contribution heatmap rendered in the terminal via `rich`.
Each cell = one day. Color intensity = number of chunk saves/updates that day.
Shows last 52 weeks by default.

```
$ granum timeline

  Granum activity — myapp (main)

        May  Jun  Jul  Aug  Sep  Oct  Nov  Dec  Jan  Feb  Mar  Apr  May
  Mon   ░░░  ░░░  ░░░  ░░░  ▒▒▒  ░░░  ▒▒▒  ░░░  ░░░  ▒▒▒  ███  ▓▓▓  ▒▒▒
  Wed   ░░░  ░░░  ░░░  ▒▒▒  ▓▓▓  ▒▒▒  ░░░  ▒▒▒  ░░░  ░░░  ▓▓▓  ███  ▓▓▓
  Fri   ░░░  ░░░  ░░░  ░░░  ░░░  ▒▒▒  ░░░  ░░░  ▒▒▒  ░░░  ▒▒▒  ▓▓▓  ░░░

  Less ░ ▒ ▓ █ More       142 saves across 38 active days
```

Color scale: 4 levels (none / light / medium / heavy) based on daily save count relative to project max.
Hovering or clicking not supported in terminal — use `granum recent` to drill into a specific day.

### `audit`
Project memory health report. Useful to run before a long session.
```
$ granum audit

Project:      /home/erik/projects/myapp (branch: main)
Active:       42 chunks
Deprecated:   5 chunks
Stale:        3 chunks  (>7d)
Very stale:   1 chunk   (>30d)
Low value:    4 chunks  (importance 1-2, stale)

Possible duplicates (similar titles, same type):
  a3f2  "auth uses OAuth2"
  b9c1  "auth flow updated"          similarity: 0.94 — consider merging

Orphaned constraints (no related decisions found):
  f8a1  "never write to .env directly"

Recommendation: run cleanup_context on 4 low-value stale chunks.
```
```
$ granum config set stale_threshold_days 14
$ granum config set compaction_threshold 30
```

### `server`
For debugging only — Claude Code auto-starts the MCP server via `.mcp.json`.

```
$ granum server status

MCP server: running
PID:        12483
Uptime:     14m 32s
DB:         .granum/db/ (1.2 MB)

$ granum server logs
[2026-05-22 14:03:01] Loaded 42 chunks for project abc123_main
[2026-05-22 14:03:01] Indexed 3 spec files (openspec/specs/)
[2026-05-22 14:03:12] query_context: "auth flow" → 7 memory + 2 spec chunks
[2026-05-22 14:03:15] save_context: updated "auth uses OAuth2" (id: a3f2)
```

---

## Plugin Structure

```
granum/
  plugin.json
  mcp_server/
    server.py             # MCP server entrypoint
    db.py                 # ChromaDB + embedding logic
    models.py             # Chunk schema
  hooks/
    granum-log.sh          # UserPromptSubmit (logging only)
    granum-compact.sh      # Stop
    granum-reinject.sh     # SessionStart (compact)
    granum-coldstart.sh    # SessionStart (startup)
    granum-spec-sync.sh    # PostToolUse (Edit|Write)
  cli/
    main.py               # typer CLI
  CLAUDE.md
  requirements.txt
  README.md
```

---

## Data Flow

### Session start
```
1. SessionStart (startup) hook fires
2. Import .granum/chunks.ndjson into local ChromaDB
3. Clear and re-index all spec files from spec_paths as type: spec chunks
4. Check git diff of spec_paths since last_session → inject change warnings if any
5. If zero active memory chunks → cold start scan (spec_paths first, then README/entry points)
6. Reset session.log
7. Update last_session in config.json
```

### Per turn
```
1. User submits prompt
2. UserPromptSubmit hook → appends prompt to session.log
3. Claude calls query_context(current prompt)
   → memory query: top 7 scored chunks (similarity * 0.8 + freshness * 0.2, per-type quota)
   → spec query: top 3 spec chunks (similarity only, no decay)
   → Claude reads both sets (age, importance, similarity, source)
4. Claude does the work (reads files, edits, runs commands)
5. If Claude edits a spec file:
   → PostToolUse hook re-indexes that file's spec chunks immediately
   → injects conflict warning for overlapping memory chunks
   → Claude deprecates/updates affected memory chunks
6. Claude calls save_context for any decisions/preferences/constraints/file_state
   → deterministic ID upsert → auto-export to chunks.ndjson
7. Claude calls cleanup_context if stale or contradictory memory chunks found
   → auto-export to chunks.ndjson
8. Claude responds to user
9. Claude self-triggers /compact if context feels long
10. Stop hook increments tool_call_count → triggers /compact at threshold as safety net
```

### After compaction
```
1. SessionStart (compact) hook fires
2. Import .granum/chunks.ndjson into local ChromaDB
3. Re-index spec files
4. Re-inject top 5 memory chunks + top 3 spec chunks via stdout
5. Re-inject last 10 prompts from session.log as recent prompt history
```

---

## Key Design Decisions

- **Deterministic chunk IDs** — `sha256(project_id + type + normalized_title)`; embeddings for retrieval only
- **Normalized titles** — lowercase, strip punctuation, collapse whitespace; defined once, applied everywhere
- **NDJSON export** — one chunk per line; git diffs readable, conflicts line-level
- **Tombstones** — soft deletes with `deleted_at`; never resurrected on import
- **Deprecated status** — chunks marked inactive without deletion
- **Importance 1–5** — Claude-assigned; used for retrieval ranking and pruning
- **Scored retrieval** — `similarity * 0.8 + freshness * 0.2`, weighted by importance, per-type quotas
- **Project + branch scoped** — `md5(git_root + ":" + branch)`; git handles merging
- **Git as sync layer** — `.granum/chunks.ndjson` committed; no remote infra needed
- **Spec indexing** — spec files indexed as `type: spec` chunks, re-indexed on every startup; queryable but read-only; separate retrieval quota (`spec_retrieval_limit`) so spec chunks never crowd out memory chunks
- **Spec file awareness** — `granum init` detects known frameworks + user-confirmed paths; stored in config
- **Spec change detection** — git diff on `spec_paths` at startup; PostToolUse hook on spec file edits
- **Session prompt log** — `UserPromptSubmit` logs all prompts to `session.log`; injected after compaction alongside chunks so Claude knows what the user was working on; resets each new session
- **Compaction: Claude-first, counter fallback** — replace with `CLAUDE_CONTEXT_PERCENT` when available
- **Fail-open hooks** — timeout + exit 0 on failure; never blocks Claude
- **Cold start** — reads spec_paths first, falls back to README/entry points; large files chunked by section

---

## MCP Server Startup

Defined in `.mcp.json` at the plugin root:

```json
{
  "mcpServers": {
    "granum": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "${pluginDir}"
    }
  }
}
```

Claude Code starts the MCP server automatically when the plugin is enabled. No daemon, no manual startup. Claude Code pipes JSON-RPC messages to stdin/stdout.

---

## `.granum/.gitignore`

Created automatically by `granum init`:

```
# Runtime files — do not commit
session.log
tool_call_count

# Commit these
# chunks.ndjson
# config.json
```

Note: `session.log` is a runtime file (resets each session) and should not be committed.

---

## Embedding Model Migration

Model name stored in `config.json`:

```json
{
  "embedding_model": "all-MiniLM-L6-v2"
}
```

On MCP server startup, if `embedding_model` in config differs from the model currently loaded:
1. Log warning: "Embedding model changed — re-embedding all chunks"
2. Re-embed all non-deleted chunks using new model
3. Update vectors in ChromaDB
4. Update `embedding_model` in config

Ensures existing vectors are never silently incompatible with new embeddings.

---

## First-Time Install UX

On first import or cold start, if the embedding model is not cached locally:

```
Granum: Downloading embedding model (all-MiniLM-L6-v2, ~80MB)...
This only happens once.
```

Printed to stderr so it appears in Claude Code's hook output without polluting stdout context injection. Progress shown via `tqdm` or simple percentage prints.

---

## Uninstall

```
granum remove [--keep-chunks]
```

- Removes hooks from `.claude/settings.json`
- Removes MCP server entry from `.mcp.json`
- By default: exports final chunks to `.granum/chunks-export.md` (human-readable markdown) before wiping DB
- `--keep-chunks`: skips DB wipe, leaves `.granum/` intact for potential reinstall
- Does not delete `.granum/chunks.ndjson` — user must do that manually if desired

Export format:
```markdown
# Granum Export — myapp (main) — 2026-05-22

## Decisions
- **update to auth flow** (importance: 5) — auth now uses OAuth2...
- ...

## Constraints
- **never touch .env directly** (importance: 4) — ...
```

---

## Platform Support

Hooks are bash scripts. **WSL required on Windows.** This must be stated clearly in README and `granum init` output.

`granum init` checks for bash availability and warns if not found:
```
Warning: bash not found. Granum hooks require bash (WSL on Windows).
MCP server and CLI will still work, but hooks will not fire.
```

---

## MCP Server Resilience

`.mcp.json` entry includes `alwaysLoad: true`:

```json
{
  "mcpServers": {
    "granum": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "${pluginDir}",
      "alwaysLoad": true
    }
  }
}
```

`alwaysLoad: true` causes Claude Code to block on startup until Granum connects and fail loudly if it can't, rather than silently missing tools mid-session.

MCP server should catch ChromaDB lock errors and embedding OOM errors at startup and exit with a clear error message rather than hanging.

---

## Chunk Content Size Limit

`save_context` enforces a max content length of 500 tokens (approx. 2000 characters). If `content` exceeds this, it is truncated and a note appended:

```
...[truncated — full content exceeded 500 token limit]
```

Claude should be instructed in CLAUDE.md to write concise chunk content. If a chunk needs more detail, split it into multiple chunks with related titles.

Added to CLAUDE.md:
```
Keep content concise — max ~400 tokens. If more detail is needed,
split into multiple chunks with related titles.
```

---

## Python Version Requirement

Minimum: **Python 3.9+** (ChromaDB requirement).

`granum init` checks Python version and exits with a clear error if below 3.9:
```
Error: Granum requires Python 3.9+. Found: Python 3.8.10
```

Stated in README prerequisites.