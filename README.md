# granum

Persistent semantic memory for Claude Code. Replaces context-window-based memory with a local vector + graph database that survives compaction, branch switches, and new sessions.

Claude saves decisions, preferences, constraints, and file state as typed chunks. Before each response, relevant chunks are retrieved via semantic search and graph traversal and injected automatically. You get a normal Claude Code conversation — granum runs invisibly in the background.

## How it works

Each chunk has a **type**, a **title**, and **content**. Claude assigns importance (1–5) and declares relationships between chunks (depends on, supersedes, contradicts, derived from). At retrieval time, granum runs a two-phase search: cosine similarity seeds, then BFS graph expansion — so related chunks surface even when their query similarity is low.

Chunks are stored in a local [Kuzu](https://kuzudb.com) graph database and exported to `.granum/chunks.ndjson` for git portability. Memory is branch-isolated: `project_id = md5(git_root:branch)`.

## Install

```bash
pip install -e .
granum init
```

Requires Python 3.9+. First run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB).

## Setup

`granum init` detects your git root and branch, writes `.granum/config.json`, and optionally indexes spec files (SPEC.md, RATIONALE.md, etc.) as read-only context chunks.

The MCP server starts automatically via `.mcp.json` when Claude Code loads the project. No manual server management needed.

## CLI

```
Memory
  granum list                    list all chunks grouped by type
  granum search "<query>"        semantic search
  granum recent                  most recently updated chunks
  granum stale                   chunks that haven't been updated recently
  granum history <id>            version history for a chunk (SUPERSEDES chain)
  granum clear                   delete all chunks for this project

Analysis
  granum stats                   project memory statistics
  granum audit                   health report — conflicts, stale, orphans, low value
  granum drift                   check file_state chunks against actual codebase
  granum summarize [--save]      recent changes as session handoff narrative
  granum timeline [--months N]   calendar heatmap of memory activity

Visualization
  granum graph                   edge list view
  granum graph <query>           neighborhood centered on closest chunk
  granum graph --open            interactive D3 force graph in browser
  granum graph --embed           PCA vector-space scatter plot in browser

Data
  granum export                  write chunks to .granum/chunks.ndjson
  granum import                  load chunks from .granum/chunks.ndjson

Setup
  granum init                    initialize granum in current project
  granum config <key> <value>    set a config value
```

## MCP tools

Claude calls these automatically — you don't invoke them directly.

| Tool | When called |
|---|---|
| `query_context` | Start of every turn |
| `save_context` | After decisions, preferences, file changes, bugs fixed |
| `cleanup_context` | Deprecate, merge, or update existing chunks |
| `add_edge` | Declare relationships between chunks |
| `check_drift` | Session start — verifies file_state chunks against codebase |
| `get_recent_changes` | When continuing previous work |
| `save_handoff` | End of session — 2-3 sentence summary for next cold start |
| `list_chunks` | Audit stored memory |
| `get_related_chunks` | Traverse graph from a chunk |
| `reindex_specs` | After editing spec files |

## Chunk types

| Type | Used for |
|---|---|
| `decision` | Architectural and implementation choices |
| `preference` | User style, workflow, naming, tool preferences |
| `constraint` | Hard limits, version pins, gotcha behaviors |
| `file_state` | Current state of files, modules, systems |
| `spec` | Read-only — indexed from spec files, never saved by Claude |

## Retrieval

Two separate queries per turn, never pooled:

- **Memory chunks**: `similarity × 0.8 + freshness × 0.2`, weighted by importance, per-type quota max 3, top 7 total. Seeds top-5 for BFS graph expansion (depth ≤ 2).
- **Spec chunks**: similarity only, no freshness decay, top 3.

Graph traversal edge weights: `DEPENDS_ON` 0.78 · `DERIVED_FROM` 0.72 · `RELATES_TO` 0.65 · `SUPERSEDES` 0.60.

## Persistence

`.granum/chunks.ndjson` — one chunk per line, committed to git. NDJSON produces line-level diffs that auto-merge cleanly. Soft deletes only (`deleted_at` tombstones). Kuzu is the runtime store; NDJSON is the portability layer.

## Project structure

```
granum/
  mcp_server/
    server.py       MCP server (stdio, JSON-RPC)
    db.py           Kuzu graph DB + embedding logic
    models.py       Chunk schema
  hooks/
    granum-log.sh          UserPromptSubmit — prompt logging + per-turn reminders
    granum-coldstart.sh    SessionStart — cold start injection
    granum-reinject.sh     SessionStart — post-compaction reinject
  cli/
    main.py         CLI (typer + rich)
  .mcp.json
  pyproject.toml
```

## License

LGPL-2.1
