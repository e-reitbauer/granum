# Granum — Design Rationale

This document explains the *why* behind key design decisions in Granum.
Read this before implementing — several decisions look suboptimal in isolation
but are intentional. Do not "simplify" them away.

---

## The Problem Being Solved

Granum exists because of **context rot** in agentic Claude Code sessions.

In a long coding session, Claude Code accumulates chat history — every file read,
every edit, every response — until the context window fills up. When compaction
happens, earlier decisions, constraints, and architectural context get summarized
or lost. The next turn, Claude doesn't know why it made the choices it did.

The result: Claude contradicts itself, re-reads files it already processed,
forgets constraints the user stated an hour ago, and loses track of what it
changed and why.

Granum's answer: **extract what matters into a persistent semantic store, and
retrieve it on demand.** Instead of relying on chat history to carry context
forward, Claude actively saves decisions as chunks and queries them each turn.
Context rot becomes irrelevant — the important stuff lives outside the window.

---

## Why Claude Queries Itself (Not a Hook)

Early in design we considered using a `UserPromptSubmit` hook to automatically
query the vector DB and inject context before Claude sees each prompt. This would
make retrieval fully transparent to the user.

**We rejected this** because hooks cannot call MCP tools. A hook is a shell
script that runs before Claude Code processes the prompt — it has no access to
the MCP server's tools. To query ChromaDB from a hook, we'd need a separate
HTTP sidecar or a standalone script duplicating the DB logic outside the MCP
server. That's two code paths for the same operation, two places to break.

**The decision:** Claude calls `query_context` itself at the start of each turn,
guided by CLAUDE.md instructions. This keeps one code path (MCP server) and
one source of truth (ChromaDB). The `UserPromptSubmit` hook is used only for
logging prompts to `session.log` — a trivial operation that needs no DB access.

Do not move query logic into hooks.

---

## Why Deterministic IDs (Not Embedding Similarity) for Identity

It might seem natural to use embedding similarity to detect "the same chunk" —
if two chunks are semantically very similar, they're probably about the same
thing, so overwrite one with the other.

**We rejected this** because of the "similar but opposite" trap:

- `"constraint: disable Redis caching"` and `"decision: enable Redis caching"`
  have cosine similarity ~0.93 — they share almost all semantic tokens.
- A threshold-based overwrite would silently replace a constraint with its
  opposite decision, corrupting the memory store with no warning.

**The decision:** chunk identity is `sha256(project_id + type + normalized_title)`.
Deterministic, collision-free, merge-safe. Two chunks with the same type and
normalized title are always the same chunk — overwrite is intentional and
explicit. Two chunks with different titles are always different chunks —
no silent merging.

Embeddings are used **only for retrieval** (finding relevant chunks by query).
They are never used for identity (deciding whether two chunks are the "same").

Do not add similarity-based upsert logic. The deterministic ID is the design.

---

## Why NDJSON (Not a JSON Array)

`.memex/chunks.json` as a JSON array was the first instinct. A JSON array is
simpler to read and write.

**We rejected this** because of git merge behavior. A JSON array is one
continuous structure — any change anywhere in the file (a new chunk, a deleted
chunk, an updated field) causes the entire array to be rewritten. When two
developers on different branches both add chunks and merge, git sees a conflict
spanning the entire file. Resolving it means manually reconstructing valid JSON
from two conflicting versions of the whole array.

**The decision:** NDJSON (`.granum/chunks.ndjson`) — one chunk per line. Each
line is a self-contained JSON object. When two developers add chunks on different
branches, git sees two new lines added at different positions — typically
auto-mergeable with no conflict. When one developer updates a chunk, only that
one line changes in the diff. Conflicts are line-level and readable.

Do not change the format to a JSON array. The NDJSON format is chosen
specifically for git ergonomics.

---

## Why the Hook Queries the Prompt Log (Not the Raw Prompt) After Compaction

After compaction, the `SessionStart (compact)` hook re-injects context. We inject
two things: top chunks from ChromaDB, and recent prompt history from
`.granum/session.log`.

The prompt history exists because chunks tell Claude *what was decided*, but not
*what the user was trying to accomplish*. After compaction, Claude loses the
thread of the conversation — it knows facts but not intent. The last 10 prompts
from `session.log` restore that intent without re-running the full conversation.

`session.log` is reset on each new session startup (not on compaction). This means
prompt history is scoped to the current working session, not accumulated across
days. A week-old prompt is not useful after compaction — it would add noise.

Do not persist `session.log` across sessions. Do not commit it to git.

---

## Why Spec Chunks Are Re-Indexed Every Startup (Not Exported to NDJSON)

Memory chunks are exported to `.granum/chunks.ndjson` and committed to git —
they are the persistent record of decisions made during coding sessions.

Spec chunks (`type: spec`) are sourced from the project's own spec files
(`openspec/specs/`, `docs/`, etc). These files are already version-controlled by
the project. Exporting them again to `chunks.ndjson` would be redundant —
two copies of the same information, one of which can go stale.

**The decision:** spec chunks are never exported. They are re-indexed from source
files on every startup. If a spec file changes, the next session automatically
picks up the change. No sync required, no stale copies.

The downside: re-indexing takes time on startup. This is acceptable because:
1. Startup only happens once per session
2. Spec files are typically small and change infrequently
3. The alternative (stale spec data) is worse than the cost

Do not add spec chunks to the NDJSON export.

---

## Why Separate Retrieval Quotas for Spec and Memory Chunks

`query_context` runs two separate queries — one for memory chunks, one for spec
chunks — and combines them with independent limits (`memory_retrieval_limit` and
`spec_retrieval_limit`).

A shared pool would mean: on a well-documented project with rich spec content,
spec chunks dominate every retrieval because spec files are typically more
comprehensive and semantically rich than short decision chunks. Memory chunks —
the implementation decisions that are Granum's core value — would rarely surface.

**The decision:** separate quotas ensure both types are always represented.
Default: 7 memory + 3 spec = 10 total. Configurable.

Do not merge the two queries into one pool.

---

## Why `granum server` Commands Are Debug-Only

Claude Code automatically starts the MCP server as a subprocess via `.mcp.json`
when the plugin is active. The server lifecycle is fully managed by Claude Code.

`granum server start/stop/status/logs` exist only so developers can inspect or
debug the server when something goes wrong — a crash, a lock, a startup failure.
They are not part of normal usage.

Do not design the system to require manual server management. If a user needs
to run `granum server start` in normal usage, something is broken.

---

## Why `importance` Is Claude-Assigned (Not Computed)

Importance (1–5) is set by Claude when saving a chunk, not computed from usage
patterns or heuristics.

We considered computing importance from: how often a chunk is retrieved, how
high its similarity scores are, how recently it was accessed. This would make
importance "objective."

**We rejected this** because importance in this context is about *what matters
for the project*, not *what gets retrieved often*. A critical architectural
constraint might never be retrieved (because it's never violated) but is extremely
important to keep. A frequently retrieved chunk about a minor preference is not
necessarily important.

Claude, having just made or observed a decision, is better positioned to judge
its importance than a retrieval frequency counter.

Do not replace Claude-assigned importance with computed metrics.