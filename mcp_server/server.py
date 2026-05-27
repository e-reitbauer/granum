from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import InitializedNotification, TextContent, Tool

from .db import GranumDB
from .ipc import start_ipc_server
from .models import ChunkType

# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

def _find_granum_dir() -> Path:
    cwd = Path(os.environ.get("GRANUM_CWD", os.getcwd()))
    return cwd / ".granum"


def _load_config(granum_dir: Path) -> dict:
    config_path = granum_dir / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            return json.load(f)
    return {}


def _make_project_id(git_root: Optional[str], branch: Optional[str]) -> str:
    git_root = git_root or os.getcwd()
    branch = branch or "main"
    raw = f"{git_root}:{branch}"
    return hashlib.md5(raw.encode()).hexdigest()


def _resolve_project_id() -> str:
    """Compute project_id from live git state. Falls back to config value."""
    import subprocess
    cwd = Path(os.environ.get("GRANUM_CWD", os.getcwd()))
    try:
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd), capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(cwd), capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        if git_root and branch:
            return _make_project_id(git_root, branch)
    except Exception:
        pass
    return config.get("project_id") or _make_project_id(None, None)


# ------------------------------------------------------------------
# Server init
# ------------------------------------------------------------------

granum_dir = _find_granum_dir()
config = _load_config(granum_dir)

project_id: str = config.get("project_id") or _make_project_id(
    os.environ.get("GRANUM_GIT_ROOT"),
    os.environ.get("GRANUM_GIT_BRANCH"),
)

db = GranumDB(
    db_path=granum_dir / "kuzu.db",
    ndjson_path=granum_dir / "chunks.ndjson",
    stale_threshold_days=config.get("stale_threshold_days", 7),
)

memory_retrieval_limit: int = config.get("memory_retrieval_limit", 10)
spec_retrieval_limit: int = config.get("spec_retrieval_limit", 10)
freshness_decay_days: int = config.get("freshness_decay_days", 90)

# Import persisted chunks on startup
try:
    db.import_ndjson()
except Exception as e:
    print(f"[granum] warning: failed to import ndjson: {e}", file=sys.stderr)

# ------------------------------------------------------------------
# MCP server
# ------------------------------------------------------------------

_SERVER_INSTRUCTIONS = (
    "Granum is your persistent memory layer. Follow this protocol strictly.\n\n"

    "BEFORE any other action each turn:\n"
    "  Call query_context with the user's message to retrieve relevant memory.\n"
    "  query_context returns {\"chunks\": [...], \"unresolved_conflicts\": [...]}.\n"
    "  If unresolved_conflicts is non-empty: resolve them FIRST before answering — call add_edge(CONTRADICTS)\n"
    "  to confirm, or cleanup_context(deprecate/merge) to resolve. Do not ignore conflicts.\n"
    "  Before acting on any retrieved chunk naming a specific file/function/flag —\n"
    "  verify it still exists in the codebase first. If not, update or deprecate before proceeding.\n\n"
    "ANYTIME mid-turn you are about to make an assumption about the project — check first:\n"
    "  If you are unsure about a past decision, a file's current state, a user preference, or a constraint,\n"
    "  call query_context with a focused query before guessing. You may have already saved the answer.\n\n"

    "SAVE TRIGGERS — save_context is REQUIRED after any turn where:\n"
    "  - You chose a tech stack, library, framework, or tool\n"
    "  - You created, renamed, moved, or deleted a file or module\n"
    "  - You made an architectural decision (structure, patterns, APIs, data models)\n"
    "  - The user stated a preference about style, workflow, naming, or behavior\n"
    "  - The user accepted a non-obvious approach without pushback — that silence is confirmation, save it\n"
    "  - You discovered a constraint (env limitation, dependency conflict, version pin)\n"
    "  - You observed code or behavior that contradicts an existing spec chunk or memory chunk (save the discrepancy, importance 4+)\n"
    "  - You encountered a surprising or non-obvious behavior that would trip up a future session\n"
    "  - You fixed a non-trivial bug (save what the bug was and how it was fixed)\n"
    "  - You completed a feature or significant unit of work\n"
    "  - The user corrected you or changed direction\n\n"

    "Save multiple chunks per turn if multiple things happened. Be specific in titles.\n"
    "Bad title: 'user preference'. Good title: 'user prefers tabs over spaces in Python files'.\n\n"

    "Chunk types:\n"
    "  decision    — architectural or implementation choices\n"
    "  preference  — user style, workflow, naming, or tool preferences\n"
    "  constraint  — hard limits, env requirements, version pins, rules, AND surprising/gotcha behaviors\n"
    "                (e.g. 'calling X before Y crashes the process' belongs here)\n"
    "  file_state  — current state of files, modules, or systems\n\n"

    "Importance scale:\n"
    "  5 — architectural, affects all future work (e.g. chosen DB, auth strategy)\n"
    "  4 — significant, affects a subsystem or workflow\n"
    "  3 — default, useful context\n"
    "  2 — minor, narrow scope\n"
    "  1 — cosmetic or rarely relevant\n\n"

    "AUDIT: when the user reports a wrong answer, or when you act on a chunk naming a specific file/function/flag —\n"
    "  call list_chunks, verify stale entries, then cleanup_context to deprecate or merge.\n"
    "After editing a spec file: call reindex_specs so changes are immediately searchable.\n\n"
    "GRAPH EDGES — required every time you call save_context:\n"
    "  After every save_context call, immediately call add_edge for each relationship you know:\n"
    "  - New chunk SUPERSEDES an older one on the same topic → add_edge(new_id, old_id, SUPERSEDES)\n"
    "  - New chunk DEPENDS_ON another (can't be understood without it) → add_edge(new_id, dep_id, DEPENDS_ON)\n"
    "  - New chunk CONTRADICTS an existing one (mutually exclusive) → add_edge(new_id, conflict_id, CONTRADICTS)\n"
    "  - New chunk was DERIVED_FROM merging others → add_edge(new_id, source_id, DERIVED_FROM)\n"
    "  Do not skip this step. No edges = no graph traversal = worse retrieval next session.\n"
    "  RELATES_TO is auto-detected by embedding similarity — never declare it manually.\n"
    "  Check the 'conflicts' field on query_context results — those are existing CONTRADICTS pairs to act on.\n\n"
    "GRAPH RAG — query_context now uses graph traversal in addition to similarity:\n"
    "  Phase 1: similarity seeds all chunks above 0.3 threshold.\n"
    "  Phase 2: BFS from top-5 memory seeds over DEPENDS_ON, RELATES_TO, DERIVED_FROM, SUPERSEDES (depth ≤ 2).\n"
    "  Phase 3: direct similarity hits override graph hits; type quotas applied; top results returned.\n"
    "  Each chunk has 'retrieved_via': 'similarity' for direct hits, or a traversal path string for graph hits.\n"
    "  Graph hits surface related chunks even when their query similarity is low — trust them.\n\n"
    "AFTER save_context: check the response for 'merge_candidates'.\n"
    "  If present, these are active chunks with cosine similarity ≥ 0.85 to the one you just saved.\n"
    "  Strong signal they cover the same topic. Consider calling cleanup_context(merge) to consolidate.\n"
    "  Each chunk also has 'retrieval_count' — how many times it has been retrieved. High count + low\n"
    "  importance = consider bumping importance. Zero count after many sessions = candidate for cleanup."
)

app = Server("granum", instructions=_SERVER_INSTRUCTIONS)


_EMBEDDING_VERSION = "2"  # bump when embedding strategy changes to force re-embed


async def _on_initialized(notification: InitializedNotification) -> None:
    """Refresh project_id from live git state and re-index specs each session."""
    global project_id
    try:
        project_id = _resolve_project_id()
        db.import_ndjson()
        # Re-embed all chunks if embedding strategy changed
        if config.get("embedding_version") != _EMBEDDING_VERSION:
            db.reembed_all(project_id)
            config["embedding_version"] = _EMBEDDING_VERSION
            (granum_dir / "config.json").write_text(json.dumps(config, indent=2))
        _reindex_all_specs()
        db.sync_spec_edges(project_id)
    except Exception as e:
        print(f"[granum] session init error: {e}", file=sys.stderr)

app.notification_handlers[InitializedNotification] = _on_initialized


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_context",
            description=(
                "Retrieve relevant context chunks from Granum memory. "
                "Call this at the start of every turn before doing any work."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query (usually the user's prompt)"},
                    "type_filter": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["decision", "preference", "file_state", "constraint", "spec"]},
                        "description": "Optional: filter to specific chunk types",
                    },
                    "memory_limit": {"type": "integer", "description": "Max memory chunks to return (default 7)"},
                    "spec_limit": {"type": "integer", "description": "Max spec chunks to return (default 3)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="save_context",
            description=(
                "Save a memory chunk. Call this after EVERY turn where you: chose a tech/library/tool, "
                "created or changed files, made an architectural decision, learned a user preference, "
                "discovered a constraint, fixed a bug, completed a feature, or were corrected. "
                "Save multiple chunks per turn if multiple things happened. Be specific in titles."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short, specific, lowercase title"},
                    "content": {"type": "string", "description": "Details — max ~400 tokens"},
                    "type": {
                        "type": "string",
                        "enum": ["decision", "preference", "file_state", "constraint"],
                    },
                    "importance": {"type": "integer", "minimum": 1, "maximum": 5, "description": "1–5; architectural=5, minor pref=1, default=3"},
                },
                "required": ["title", "content", "type"],
            },
        ),
        Tool(
            name="cleanup_context",
            description="Delete, deprecate, merge, or update existing memory chunks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["delete", "deprecate", "merge", "update"]},
                    "chunk_ids": {"type": "array", "items": {"type": "string"}},
                    "merged_title": {"type": "string"},
                    "merged_content": {"type": "string"},
                    "merged_type": {"type": "string", "enum": ["decision", "preference", "file_state", "constraint"]},
                    "merged_importance": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["action", "chunk_ids"],
            },
        ),
        Tool(
            name="list_chunks",
            description=(
                "List all memory chunks for this project. Use this to audit stored memory, "
                "find chunks to clean up, or get chunk IDs before calling cleanup_context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "include_deprecated": {"type": "boolean", "description": "Include deprecated chunks (default false)"},
                    "type_filter": {
                        "type": "string",
                        "enum": ["decision", "preference", "file_state", "constraint"],
                        "description": "Optional: filter to one chunk type",
                    },
                },
            },
        ),
        Tool(
            name="get_chunk",
            description="Get full content of a specific memory chunk by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                },
                "required": ["chunk_id"],
            },
        ),
        Tool(
            name="reindex_specs",
            description=(
                "Re-index all spec files from configured spec_paths. "
                "Call this after editing or creating a spec file so the changes are immediately searchable."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="add_edge",
            description=(
                "Declare a relationship between two memory chunks. Use when you understand "
                "a relationship that auto-detection can't infer: "
                "SUPERSEDES (A replaces B), DEPENDS_ON (A only makes sense given B), "
                "CONTRADICTS (A and B are mutually exclusive), DERIVED_FROM (A was merged from B)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "from_id":   {"type": "string", "description": "ID of the source chunk"},
                    "to_id":     {"type": "string", "description": "ID of the target chunk"},
                    "edge_type": {"type": "string", "enum": list(["CONTRADICTS", "SUPERSEDES", "RELATES_TO", "DERIVED_FROM", "DEPENDS_ON"])},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Confidence 0–1, default 1.0"},
                },
                "required": ["from_id", "to_id", "edge_type"],
            },
        ),
        Tool(
            name="get_related_chunks",
            description=(
                "Traverse the memory graph from a chunk. Returns neighbors with edge types "
                "(CONTRADICTS, SUPERSEDES, RELATES_TO, DERIVED_FROM, DEPENDS_ON). "
                "Use when you need the full context chain around a decision, or to audit conflicts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id":  {"type": "string", "description": "ID of the chunk to start from"},
                    "edge_type": {"type": "string", "enum": ["CONTRADICTS", "SUPERSEDES", "RELATES_TO", "DERIVED_FROM", "DEPENDS_ON"], "description": "Optional: filter to one edge type"},
                    "depth":     {"type": "integer", "minimum": 1, "maximum": 2, "description": "Hop depth (default 1)"},
                },
                "required": ["chunk_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "query_context":
            result = db.query_context(
                project_id=project_id,
                query=arguments["query"],
                type_filter=arguments.get("type_filter"),
                memory_limit=arguments.get("memory_limit", memory_retrieval_limit),
                spec_limit=arguments.get("spec_limit", spec_retrieval_limit),
                freshness_decay_days=freshness_decay_days,
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "save_context":
            result = db.save_context(
                project_id=project_id,
                title=arguments["title"],
                content=arguments["content"],
                chunk_type=arguments["type"],
                importance=arguments.get("importance", 3),
            )
            db.export_ndjson(project_id)
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "cleanup_context":
            result = db.cleanup_context(
                action=arguments["action"],
                chunk_ids=arguments["chunk_ids"],
                merged_title=arguments.get("merged_title"),
                merged_content=arguments.get("merged_content"),
                merged_type=arguments.get("merged_type"),
                merged_importance=arguments.get("merged_importance"),
                project_id=project_id,
            )
            db.export_ndjson(project_id)
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "list_chunks":
            chunks = db.get_all_memory_chunks(
                project_id,
                include_deprecated=arguments.get("include_deprecated", False),
            )
            type_filter = arguments.get("type_filter")
            if type_filter:
                chunks = [c for c in chunks if c.type == type_filter]
            result = [
                {"id": c.id, "type": c.type, "title": c.title,
                 "importance": c.importance, "status": c.status, "updated_at": c.updated_at}
                for c in chunks
            ]
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_chunk":
            chunk = db._get_by_id(arguments["chunk_id"])
            if chunk is None:
                return [TextContent(type="text", text=json.dumps({"error": "chunk not found"}))]
            return [TextContent(type="text", text=json.dumps(chunk.to_dict(), indent=2))]

        elif name == "reindex_specs":
            count = _reindex_all_specs()
            return [TextContent(type="text", text=json.dumps({"indexed": count}))]

        elif name == "add_edge":
            result = db.add_edge(
                from_id=arguments["from_id"],
                to_id=arguments["to_id"],
                edge_type=arguments["edge_type"],
                confidence=arguments.get("confidence", 1.0),
            )
            db.export_ndjson(project_id)
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "get_related_chunks":
            edges = db.get_edges(
                chunk_id=arguments["chunk_id"],
                edge_type=arguments.get("edge_type"),
                depth=arguments.get("depth", 1),
            )
            return [TextContent(type="text", text=json.dumps(edges, indent=2))]

        else:
            return [TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


# ------------------------------------------------------------------
# Spec indexing helpers (used by IPC handlers — single ChromaDB client)
# ------------------------------------------------------------------

_MAX_CHUNK_CHARS = 1800  # split sections larger than this by paragraph


def _split_by_paragraph(title: str, content: str, source: str) -> list[dict]:
    """Split a large section into paragraph-sized chunks."""
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    chunks = []
    current = ""
    part = 1
    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) > _MAX_CHUNK_CHARS and current:
            chunks.append({"title": f"{title} ({part})", "content": current, "source": source})
            current = para
            part += 1
        else:
            current = candidate
    if current:
        label = f"{title} ({part})" if part > 1 else title
        chunks.append({"title": label, "content": current, "source": source})
    return chunks


def _chunk_by_section(text: str, source_file: str) -> list[dict]:
    """Split a spec file into heading-scoped chunks, further splitting large sections."""
    from pathlib import Path as _Path
    file_stem = _Path(source_file).stem

    # Parse heading hierarchy and accumulate body lines
    sections: list[tuple[str, list[str]]] = []  # (heading_path, lines)
    heading_stack: list[tuple[int, str]] = []   # (level, text)
    current_lines: list[str] = []

    def _flush(stack: list[tuple[int, str]], lines: list[str]) -> None:
        if not stack:
            title = file_stem
        else:
            title = " > ".join(h for _, h in stack)
        sections.append((title, list(lines)))

    for line in text.splitlines():
        if line.startswith("#"):
            _flush(heading_stack, current_lines)
            current_lines = []
            level = len(line) - len(line.lstrip("#"))
            heading_text = line.lstrip("#").strip()
            # Pop stack to current level
            heading_stack = [(l, h) for l, h in heading_stack if l < level]
            heading_stack.append((level, heading_text))
        else:
            current_lines.append(line)

    _flush(heading_stack, current_lines)

    # Build chunks, splitting oversized sections by paragraph
    chunks: list[dict] = []
    for title, lines in sections:
        content = "\n".join(lines).strip()
        if not content:
            continue
        source = f"{source_file}#{title}"
        if len(content) <= _MAX_CHUNK_CHARS:
            chunks.append({"title": title, "content": content, "source": source})
        else:
            chunks.extend(_split_by_paragraph(title, content, source))

    return chunks if chunks else [{"title": file_stem, "content": text[:_MAX_CHUNK_CHARS], "source": source_file}]


def _reindex_all_specs() -> int:
    cwd = Path(os.environ.get("GRANUM_CWD", os.getcwd()))
    spec_paths = config.get("spec_paths", [])
    db.clear_spec_chunks(project_id)
    count = 0
    for spec_path in spec_paths:
        full_path = cwd / spec_path
        if not full_path.exists():
            continue
        files = list(full_path.rglob("*.md")) if full_path.is_dir() else [full_path]
        for f in files:
            try:
                text = f.read_text(errors="replace")
                rel = str(f.relative_to(cwd))
                db.index_spec_file(project_id, rel, _chunk_by_section(text, rel))
                count += 1
            except Exception:
                pass
    return count


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

async def _ipc_handler(method: str, params: dict):
    if method == "get_edges":
        return db.get_edges(
            params["chunk_id"],
            edge_type=params.get("edge_type"),
            depth=params.get("depth", 1),
        )
    if method == "get_all_edges":
        return db.get_all_edges(params["project_id"])
    if method == "get_chunk_history":
        return db.get_chunk_history(params["chunk_id"])
    if method == "list_chunks":
        chunks = db.get_all_memory_chunks(
            params["project_id"],
            include_deprecated=params.get("include_deprecated", False),
        )
        return [c.to_dict() for c in chunks]
    if method == "list_spec_chunks":
        chunks = db.get_spec_chunks(params["project_id"])
        return [c.to_dict() for c in chunks]
    if method == "reindex_specs":
        count = _reindex_all_specs()
        db.sync_spec_edges(project_id)
        return {"indexed": count}
    if method == "reindex_spec_file":
        file_path = params["file_path"]
        rel_path = params["rel_path"]
        try:
            text = Path(file_path).read_text(errors="replace")
            chunks = _chunk_by_section(text, rel_path)
            db.index_spec_file(project_id, rel_path, chunks)
            return {"indexed": len(chunks)}
        except Exception as e:
            return {"error": str(e)}
    if method == "coldstart_tasks":
        import subprocess as _sp
        cwd = Path(os.environ.get("GRANUM_CWD", os.getcwd()))
        spec_paths = config.get("spec_paths", [])
        last_session = config.get("last_session")

        # Check spec diffs since last session
        changed: list[str] = []
        if last_session and spec_paths:
            for sp in spec_paths:
                try:
                    r = _sp.run(
                        ["git", "diff", "--name-only", last_session, "--", sp],
                        cwd=str(cwd), capture_output=True, text=True, timeout=5,
                    )
                    changed.extend(l for l in r.stdout.strip().splitlines() if l)
                except Exception:
                    pass

        # Cold start: no active chunks → save marker
        if not db.get_all_memory_chunks(project_id):
            sources = [sp for sp in spec_paths if (cwd / sp).exists()]
            if not sources and (cwd / "README.md").exists():
                sources = ["README.md"]
            if sources:
                db.save_context(
                    project_id=project_id,
                    title="cold start — spec sources indexed",
                    content=f"Indexed on cold start: {', '.join(sources)}. Query context for details.",
                    chunk_type="file_state",
                    importance=2,
                )
                db.export_ndjson(project_id)

        # Update last_session in config
        from datetime import datetime as _dt, timezone as _tz
        config["last_session"] = _dt.now(_tz.utc).isoformat()
        (granum_dir / "config.json").write_text(json.dumps(config, indent=2))

        return {"changed_specs": changed}
    if method == "query_context":
        return db.query_context(
            project_id=project_id,
            query=params["query"],
            type_filter=params.get("type_filter"),
            memory_limit=params.get("memory_limit", memory_retrieval_limit),
            spec_limit=params.get("spec_limit", spec_retrieval_limit),
            freshness_decay_days=freshness_decay_days,
        )
    elif method == "save_context":
        result = db.save_context(
            project_id=project_id,
            title=params["title"],
            content=params["content"],
            chunk_type=params["type"],
            importance=params.get("importance", 3),
        )
        db.export_ndjson(project_id)
        return result
    elif method == "cleanup_context":
        result = db.cleanup_context(
            action=params["action"],
            chunk_ids=params["chunk_ids"],
            merged_title=params.get("merged_title"),
            merged_content=params.get("merged_content"),
            merged_type=params.get("merged_type"),
            merged_importance=params.get("merged_importance"),
            project_id=project_id,
        )
        db.export_ndjson(project_id)
        return result
    else:
        raise ValueError(f"unknown method: {method}")


async def main() -> None:
    import asyncio
    ipc_task = asyncio.create_task(start_ipc_server(_ipc_handler, granum_dir))
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    finally:
        ipc_task.cancel()
        from .ipc import socket_path
        socket_path(granum_dir).unlink(missing_ok=True)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
