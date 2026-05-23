# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e .             # install with CLI entrypoint
granum init                  # first-time setup
granum list                  # list all chunks
granum search "<query>"      # semantic search
granum stats                 # project stats
granum audit                 # health report
python -m mcp_server.server  # run MCP server directly (debug)
```

## Project

Granum is a Claude Code plugin that replaces context-based memory with a persistent semantic vector database. Claude saves decisions/preferences/state as chunks; before each response, relevant chunks are retrieved and injected. The user sees a normal Claude Code conversation.

**Status:** Pre-implementation. SPEC.md and RATIONALE.md define the full design. Read both before implementing anything.

## Tech Stack

- MCP server: Python, `mcp` SDK
- Vector store: ChromaDB (local, on-disk), `metric="cosine"` explicitly
- Embeddings: `sentence-transformers` — `all-MiniLM-L6-v2` (384d)
- CLI: `typer` + `rich`
- Hooks: Bash scripts
- Python 3.9+ required (ChromaDB requirement)

## Architecture

### Plugin structure

```
granum/
  plugin.json
  mcp_server/
    server.py        # MCP server entrypoint (stdio, JSON-RPC)
    db.py            # ChromaDB + embedding logic
    models.py        # Chunk schema
  hooks/
    granum-log.sh          # UserPromptSubmit — logging only
    granum-compact.sh      # Stop — compaction counter
    granum-reinject.sh     # SessionStart (compact matcher)
    granum-coldstart.sh    # SessionStart (startup matcher)
    granum-spec-sync.sh    # PostToolUse (Edit|Write matcher)
  cli/
    main.py          # typer CLI
  .mcp.json
  CLAUDE.md
  requirements.txt
```

### Data flow

**Session start:** import `chunks.ndjson` → re-index spec files → check git diff of spec_paths → cold start scan if zero active chunks → reset `session.log` → update `last_session`.

**Per turn:** Claude calls `query_context` first → does work → calls `save_context`/`cleanup_context` → auto-exports to `chunks.ndjson`.

**After compaction:** re-import chunks → re-index specs → re-inject top 5 memory + top 3 spec chunks + last 10 prompts from `session.log`.

### Chunk identity

ID = `sha256(project_id + type + normalized_title)`. Deterministic — embeddings are for retrieval only, never identity. Same title+type = upsert. Title normalization: lowercase, strip punctuation, collapse whitespace.

### Project scoping

`project_id` = `md5(git_root_path + ":" + git_branch)`. Branch-isolated.

### Retrieval scoring

Memory chunks: `similarity * 0.8 + freshness * 0.2`, weighted by importance, per-type quota max 3, top 7 total. Spec chunks: similarity only (no freshness decay), top 3. Two separate queries — never a shared pool.

### Spec chunks

`type: spec` — read-only, sourced from spec files. Re-indexed every startup from `spec_paths`. Never exported to `chunks.ndjson`. `save_context` rejects `type: spec`.

### Persistence

`.granum/chunks.ndjson` — NDJSON (one chunk per line) committed to git. JSON array format was rejected: NDJSON produces line-level diffs, git auto-merges cleanly. Soft deletes only (`deleted_at` tombstones, never resurrected on import).

### Hooks architecture

Hooks are bash scripts — they cannot call MCP tools. Query logic lives only in the MCP server. `UserPromptSubmit` hook logs prompts only. All hooks: timeout protection, fail-open (exit 0 on error).

## Key Invariants (Do Not Simplify Away)

- **Deterministic IDs** — do not add similarity-based upsert. `"disable Redis caching"` and `"enable Redis caching"` have cosine similarity ~0.93 — similarity-based identity would corrupt the store.
- **NDJSON** — do not change to JSON array.
- **Spec chunks not exported** — re-indexed from source each startup. No NDJSON export for spec type.
- **Separate retrieval quotas** — do not merge spec and memory queries into one pool.
- **Query in CLAUDE.md, not hooks** — hooks cannot call MCP tools; Claude self-queries each turn.
- **Importance is Claude-assigned** — do not compute from retrieval frequency.
- **`session.log` resets each session** — not on compaction, not committed to git.

## CLI Style

- `rich` for all output. No plain `print`.
- Accent color: `#da7756` (Claude brand terra cotta orange) for headers, progress bars, highlighted values.
- Icons: Unicode only (`✓ ✗ ⚠ · ◆ ▲ ★ ▪ ◇ ○ × → ─`). No emoji, no Nerd Fonts.
- Every DB operation shows a spinner (`dots` style, `#da7756`) or progress bar.
- Destructive actions require confirmation with orange warning text.
- Error messages always explain what failed and what to do next.

## MCP Server

Started automatically by Claude Code via `.mcp.json` with `alwaysLoad: true` — blocks on startup, fails loudly if it can't connect. `granum server` subcommands are debug-only.

Chunk content max: 500 tokens (~2000 chars). Truncate with `...[truncated — full content exceeded 500 token limit]`.

On startup: if `embedding_model` in config differs from loaded model, re-embed all non-deleted chunks before serving.
