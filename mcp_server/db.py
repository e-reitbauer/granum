from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import kuzu

from .models import (
    MEMORY_TYPES,
    Chunk,
    ChunkType,
    make_chunk_id,
    make_embedding_target,
    normalize_title,
)

MAX_CONTENT_CHARS = 2000
TRUNCATION_SUFFIX = "...[truncated — full content exceeded 500 token limit]"

EDGE_TYPES = ("CONTRADICTS", "SUPERSEDES", "RELATES_TO", "DERIVED_FROM", "DEPENDS_ON")

EDGE_COLORS = {
    "CONTRADICTS":  "conflict",
    "SUPERSEDES":   "replaces",
    "RELATES_TO":   "related",
    "DERIVED_FROM": "merged from",
    "DEPENDS_ON":   "depends on",
}

_OPPOSITE_PAIRS = [
    ("enable", "disable"), ("use", "avoid"), ("add", "remove"),
    ("prefer", "avoid"), ("always", "never"), ("sync", "async"),
    ("keep", "remove"), ("allow", "deny"), ("on", "off"),
    ("start", "stop"), ("create", "delete"), ("include", "exclude"),
]

_model = None


def _get_model():
    global _model
    if _model is None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        with open(os.devnull, "w") as devnull:
            old = sys.stderr
            sys.stderr = devnull
            try:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer("all-MiniLM-L6-v2")
            finally:
                sys.stderr = old
    return _model


def _embed(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()


def _dot(a: list[float], b: list[float]) -> float:
    # Vectors are unit-normalized so dot product == cosine similarity
    return sum(x * y for x, y in zip(a, b))


def _truncate_content(content: str) -> str:
    if len(content) <= MAX_CONTENT_CHARS:
        return content
    cutoff = MAX_CONTENT_CHARS - len(TRUNCATION_SUFFIX)
    return content[:cutoff] + TRUNCATION_SUFFIX


def _age_str(updated_at: str) -> str:
    try:
        then = datetime.fromisoformat(updated_at)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - then
        days = delta.days
        if days == 0:
            hours = delta.seconds // 3600
            return f"{hours}h ago" if hours > 0 else "just now"
        return "1 day ago" if days == 1 else f"{days} days ago"
    except Exception:
        return "unknown"


def _freshness_score(updated_at: str, decay_days: int = 90) -> float:
    try:
        then = datetime.fromisoformat(updated_at)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - then).total_seconds() / 86400
        return max(0.0, 1.0 - age_days / decay_days)
    except Exception:
        return 0.0


def _is_contradicting(title_a: str, title_b: str) -> bool:
    words_a = set(normalize_title(title_a).split())
    words_b = set(normalize_title(title_b).split())
    if len(words_a & words_b) < 2:
        return False
    for w1, w2 in _OPPOSITE_PAIRS:
        if (w1 in words_a and w2 in words_b) or (w2 in words_a and w1 in words_b):
            return True
    return False


# ------------------------------------------------------------------
# GranumDB
# ------------------------------------------------------------------

class GranumDB:
    def __init__(self, db_path: Path, ndjson_path: Path, stale_threshold_days: int = 7):
        self.db_path = db_path
        self.ndjson_path = ndjson_path
        self.edges_ndjson_path = ndjson_path.parent / "edges.ndjson"
        self.stale_threshold_days = stale_threshold_days

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(str(db_path))
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Chunk(
                id         STRING,
                project_id STRING,
                title      STRING,
                content    STRING,
                type       STRING,
                importance INT64,
                status     STRING,
                source     STRING,
                embedding  DOUBLE[],
                created_at STRING,
                updated_at STRING,
                deleted_at STRING,
                PRIMARY KEY(id)
            )
        """)
        for et in EDGE_TYPES:
            self._conn.execute(f"""
                CREATE REL TABLE IF NOT EXISTS {et}(
                    FROM Chunk TO Chunk,
                    confidence DOUBLE,
                    created_by STRING,
                    created_at STRING
                )
            """)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _CHUNK_COLS = [
        "id", "project_id", "title", "content", "type",
        "importance", "status", "source", "embedding",
        "created_at", "updated_at", "deleted_at",
    ]

    _CHUNK_RETURN = ", ".join(f"c.{col}" for col in _CHUNK_COLS)

    def _row_to_chunk(self, row: list) -> Chunk:
        d = dict(zip(self._CHUNK_COLS, row))
        return Chunk(
            id=d["id"],
            project_id=d["project_id"],
            title=d["title"],
            content=d["content"],
            type=d["type"],
            importance=int(d.get("importance") or 3),
            status=d.get("status") or "active",
            source=d.get("source") or None,
            created_at=d.get("created_at") or "",
            updated_at=d.get("updated_at") or "",
            deleted_at=d.get("deleted_at") or None,
        )

    def _get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        res = self._conn.execute(
            f"MATCH (c:Chunk {{id: $id}}) RETURN {self._CHUNK_RETURN}",
            {"id": chunk_id},
        )
        return self._row_to_chunk(res.get_next()) if res.has_next() else None

    def _exists(self, chunk_id: str) -> bool:
        res = self._conn.execute(
            "MATCH (c:Chunk {id: $id}) RETURN count(c)", {"id": chunk_id}
        )
        return res.get_next()[0] > 0

    def _upsert_chunk(self, chunk: Chunk, embedding: Optional[list[float]] = None) -> None:
        if embedding is None:
            embedding = _embed(make_embedding_target(chunk.type, chunk.title, chunk.content))
        p = {
            "id":         chunk.id,
            "project_id": chunk.project_id,
            "title":      chunk.title,
            "content":    chunk.content,
            "type":       chunk.type,
            "importance": int(chunk.importance),
            "status":     chunk.status,
            "source":     chunk.source or "",
            "embedding":  embedding,
            "created_at": chunk.created_at,
            "updated_at": chunk.updated_at,
            "deleted_at": chunk.deleted_at or "",
        }
        if self._exists(chunk.id):
            self._conn.execute("""
                MATCH (c:Chunk {id: $id})
                SET c.title      = $title,
                    c.content    = $content,
                    c.type       = $type,
                    c.importance = $importance,
                    c.status     = $status,
                    c.source     = $source,
                    c.embedding  = $embedding,
                    c.updated_at = $updated_at,
                    c.deleted_at = $deleted_at
            """, p)
        else:
            self._conn.execute("""
                CREATE (:Chunk {
                    id: $id, project_id: $project_id, title: $title,
                    content: $content, type: $type, importance: $importance,
                    status: $status, source: $source, embedding: $embedding,
                    created_at: $created_at, updated_at: $updated_at, deleted_at: $deleted_at
                })
            """, p)

    def _edge_exists(self, from_id: str, to_id: str, edge_type: str) -> bool:
        res = self._conn.execute(
            f"MATCH (a:Chunk {{id: $a}})-[:{edge_type}]->(b:Chunk {{id: $b}}) RETURN count(*)",
            {"a": from_id, "b": to_id},
        )
        return res.get_next()[0] > 0

    def _add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        confidence: float = 1.0,
        created_by: str = "auto",
    ) -> None:
        if self._edge_exists(from_id, to_id, edge_type):
            return
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            f"""
            MATCH (a:Chunk {{id: $a}}), (b:Chunk {{id: $b}})
            CREATE (a)-[:{edge_type} {{
                confidence: $conf, created_by: $by, created_at: $at
            }}]->(b)
            """,
            {"a": from_id, "b": to_id, "conf": confidence, "by": created_by, "at": now},
        )

    def _auto_detect_edges(self, chunk: Chunk, embedding: list[float]) -> None:
        """Detect CONTRADICTS / RELATES_TO edges after saving a chunk."""
        res = self._conn.execute(
            """
            MATCH (c:Chunk)
            WHERE c.project_id = $pid
              AND c.status = 'active'
              AND c.deleted_at = ''
              AND c.id <> $id
            RETURN c.id, c.title, c.type, c.embedding
            """,
            {"pid": chunk.project_id, "id": chunk.id},
        )
        candidates = []
        while res.has_next():
            row = res.get_next()
            if row[3]:
                sim = _dot(embedding, row[3])
                candidates.append({"id": row[0], "title": row[1], "type": row[2], "sim": sim})

        candidates.sort(key=lambda x: x["sim"], reverse=True)
        for c in candidates[:10]:
            sim = c["sim"]
            cross_type = (chunk.type == "spec") != (c["type"] == "spec")
            relates_threshold = 0.60 if cross_type else 0.72
            # CONTRADICTS: high similarity + either keyword pair OR direct negation signal
            if sim >= 0.88 and _is_contradicting(chunk.title, c["title"]):
                self._add_edge(chunk.id, c["id"], "CONTRADICTS", sim, "auto")
                self._add_edge(c["id"], chunk.id, "CONTRADICTS", sim, "auto")
            elif sim >= relates_threshold and not self._edge_exists(chunk.id, c["id"], "CONTRADICTS"):
                self._add_edge(chunk.id, c["id"], "RELATES_TO", sim, "auto")

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        confidence: float = 1.0,
    ) -> dict:
        """Claude-declared edge. Validates IDs and edge type."""
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"unknown edge type: {edge_type}. Must be one of {EDGE_TYPES}")
        from_chunk = self._get_by_id(from_id)
        to_chunk = self._get_by_id(to_id)
        if not from_chunk:
            raise ValueError(f"chunk not found: {from_id}")
        if not to_chunk:
            raise ValueError(f"chunk not found: {to_id}")
        # CONTRADICTS is always bidirectional
        self._add_edge(from_id, to_id, edge_type, confidence, "claude")
        if edge_type == "CONTRADICTS":
            self._add_edge(to_id, from_id, edge_type, confidence, "claude")
        return {
            "edge_type": edge_type,
            "from": from_chunk.title,
            "to": to_chunk.title,
            "confidence": confidence,
        }

    def sync_spec_edges(self, project_id: str) -> int:
        """Retroactive scan: connect all memory chunks to related spec chunks.
        Called after spec reindex so existing memory chunks get linked."""
        res = self._conn.execute(
            """
            MATCH (s:Chunk)
            WHERE s.project_id = $pid AND s.type = 'spec' AND s.deleted_at = ''
            RETURN s.id, s.title, s.embedding
            """,
            {"pid": project_id},
        )
        spec_chunks = []
        while res.has_next():
            row = res.get_next()
            if row[2]:
                spec_chunks.append({"id": row[0], "title": row[1], "embedding": row[2]})

        if not spec_chunks:
            return 0

        res = self._conn.execute(
            """
            MATCH (m:Chunk)
            WHERE m.project_id = $pid
              AND m.type IN ['decision', 'preference', 'file_state', 'constraint']
              AND m.status = 'active'
              AND m.deleted_at = ''
            RETURN m.id, m.title, m.embedding
            """,
            {"pid": project_id},
        )
        memory_chunks = []
        while res.has_next():
            row = res.get_next()
            if row[2]:
                memory_chunks.append({"id": row[0], "title": row[1], "embedding": row[2]})

        count = 0
        for m in memory_chunks:
            for s in spec_chunks:
                sim = _dot(m["embedding"], s["embedding"])
                if sim >= 0.88 and _is_contradicting(m["title"], s["title"]):
                    self._add_edge(m["id"], s["id"], "CONTRADICTS", sim, "auto")
                    self._add_edge(s["id"], m["id"], "CONTRADICTS", sim, "auto")
                    count += 1
                elif sim >= 0.60:
                    self._add_edge(m["id"], s["id"], "RELATES_TO", sim, "auto")
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Import / export
    # ------------------------------------------------------------------

    def import_ndjson(self) -> int:
        count = 0
        if self.ndjson_path.exists():
            with self.ndjson_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = Chunk.from_dict(json.loads(line))
                    except Exception:
                        continue
                    if chunk.deleted_at:
                        self._conn.execute(
                            "MATCH (c:Chunk {id: $id}) DETACH DELETE c", {"id": chunk.id}
                        )
                        continue
                    self._upsert_chunk(chunk)
                    count += 1

        if self.edges_ndjson_path.exists():
            with self.edges_ndjson_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                        self._add_edge(
                            e["from_id"], e["to_id"], e["edge_type"],
                            e.get("confidence", 1.0), e.get("created_by", "auto"),
                        )
                    except Exception:
                        continue
        return count

    def export_ndjson(self, project_id: str) -> None:
        existing: dict[str, dict] = {}
        if self.ndjson_path.exists():
            with self.ndjson_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        existing[d["id"]] = d
                    except Exception:
                        pass

        type_list = "', '".join(MEMORY_TYPES)
        res = self._conn.execute(
            f"""
            MATCH (c:Chunk)
            WHERE c.project_id = $pid AND c.type IN ['{type_list}']
            RETURN c.id, c.project_id, c.title, c.content, c.type,
                   c.importance, c.status, c.source, c.created_at, c.updated_at, c.deleted_at
            """,
            {"pid": project_id},
        )
        cols = ["id", "project_id", "title", "content", "type", "importance",
                "status", "source", "created_at", "updated_at", "deleted_at"]
        output: dict[str, dict] = {}
        live_ids: set[str] = set()

        while res.has_next():
            row = dict(zip(cols, res.get_next()))
            live_ids.add(row["id"])
            output[row["id"]] = row

        now = datetime.now(timezone.utc).isoformat()
        for chunk_id, d in existing.items():
            if d.get("project_id") != project_id:
                output.setdefault(chunk_id, d)
            elif chunk_id not in live_ids and not d.get("deleted_at"):
                output[chunk_id] = {**d, "deleted_at": now}
            elif chunk_id not in output:
                output[chunk_id] = d

        self.ndjson_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ndjson_path.open("w") as f:
            for d in output.values():
                f.write(json.dumps(d) + "\n")

        self._export_edges_ndjson(project_id)

    def _export_edges_ndjson(self, project_id: str) -> None:
        existing: dict[tuple, dict] = {}
        if self.edges_ndjson_path.exists():
            with self.edges_ndjson_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                        existing[(e["from_id"], e["to_id"], e["edge_type"])] = e
                    except Exception:
                        pass

        for et in EDGE_TYPES:
            try:
                res = self._conn.execute(
                    f"""
                    MATCH (a:Chunk)-[r:{et}]->(b:Chunk)
                    WHERE a.project_id = $pid
                    RETURN a.id, b.id, r.confidence, r.created_by, r.created_at
                    """,
                    {"pid": project_id},
                )
                while res.has_next():
                    row = res.get_next()
                    key = (row[0], row[1], et)
                    existing[key] = {
                        "from_id":    row[0],
                        "to_id":      row[1],
                        "edge_type":  et,
                        "confidence": row[2],
                        "created_by": row[3],
                        "created_at": row[4],
                    }
            except Exception:
                pass

        with self.edges_ndjson_path.open("w") as f:
            for e in existing.values():
                f.write(json.dumps(e) + "\n")

    # ------------------------------------------------------------------
    # save_context
    # ------------------------------------------------------------------

    def save_context(
        self,
        project_id: str,
        title: str,
        content: str,
        chunk_type: ChunkType,
        importance: int = 3,
    ) -> dict:
        if chunk_type == "spec":
            raise ValueError("spec chunks are read-only — written only by the indexer")

        content = _truncate_content(content)
        chunk_id = make_chunk_id(project_id, chunk_type, title)
        embedding = _embed(make_embedding_target(chunk_type, title, content))

        existing = self._get_by_id(chunk_id)
        now = datetime.now(timezone.utc).isoformat()

        if existing:
            existing.content = content
            existing.updated_at = now
            existing.importance = importance
            self._upsert_chunk(existing, embedding)
            action = "updated"
        else:
            chunk = Chunk.create(
                project_id=project_id,
                title=title,
                content=content,
                chunk_type=chunk_type,
                importance=importance,
            )
            self._upsert_chunk(chunk, embedding)
            action = "created"

        self._auto_detect_edges(
            Chunk(
                id=chunk_id, project_id=project_id, title=title,
                content=content, type=chunk_type, importance=importance,
                status="active",
            ),
            embedding,
        )
        return {"action": action, "id": chunk_id}

    # ------------------------------------------------------------------
    # cleanup_context
    # ------------------------------------------------------------------

    def cleanup_context(
        self,
        action: str,
        chunk_ids: list[str],
        merged_title: Optional[str] = None,
        merged_content: Optional[str] = None,
        merged_type: Optional[ChunkType] = None,
        merged_importance: Optional[int] = None,
        project_id: Optional[str] = None,
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()

        if action == "delete":
            for cid in chunk_ids:
                self._conn.execute(
                    "MATCH (c:Chunk {id: $id}) DETACH DELETE c", {"id": cid}
                )
            return {"action": "deleted", "ids": chunk_ids}

        elif action == "deprecate":
            for cid in chunk_ids:
                self._conn.execute(
                    "MATCH (c:Chunk {id: $id}) SET c.status = 'deprecated', c.updated_at = $now",
                    {"id": cid, "now": now},
                )
            return {"action": "deprecated", "ids": chunk_ids}

        elif action == "merge":
            if not all([merged_title, merged_content, merged_type, project_id]):
                raise ValueError("merge requires merged_title, merged_content, merged_type, project_id")
            for cid in chunk_ids:
                self._conn.execute(
                    "MATCH (c:Chunk {id: $id}) SET c.status = 'deprecated', c.updated_at = $now",
                    {"id": cid, "now": now},
                )
            result = self.save_context(
                project_id, merged_title, merged_content, merged_type, merged_importance or 3
            )
            for cid in chunk_ids:
                self._add_edge(result["id"], cid, "DERIVED_FROM", 1.0, "claude")
            return {"action": "merged", "deprecated_ids": chunk_ids, "new_id": result["id"]}

        elif action == "update":
            if len(chunk_ids) != 1:
                raise ValueError("update requires exactly one chunk_id")
            cid = chunk_ids[0]
            params: dict = {"id": cid, "now": now}
            sets = ["c.updated_at = $now"]
            if merged_content:
                params["content"] = _truncate_content(merged_content)
                sets.append("c.content = $content")
            if merged_importance:
                params["importance"] = int(merged_importance)
                sets.append("c.importance = $importance")
            self._conn.execute(
                f"MATCH (c:Chunk {{id: $id}}) SET {', '.join(sets)}", params
            )
            return {"action": "updated", "id": cid}

        else:
            raise ValueError(f"unknown action: {action}")

    # ------------------------------------------------------------------
    # query_context
    # ------------------------------------------------------------------

    def query_context(
        self,
        project_id: str,
        query: str,
        type_filter: Optional[list[ChunkType]] = None,
        memory_limit: int = 7,
        spec_limit: int = 3,
        freshness_decay_days: int = 90,
    ) -> list[dict]:
        query_emb = _embed(query)
        results = []

        memory_types = [t for t in (type_filter or list(MEMORY_TYPES)) if t != "spec"]

        if memory_types:
            type_list = "', '".join(memory_types)
            res = self._conn.execute(
                f"""
                MATCH (c:Chunk)
                WHERE c.project_id = $pid
                  AND c.status = 'active'
                  AND c.deleted_at = ''
                  AND c.type IN ['{type_list}']
                RETURN c.id, c.title, c.content, c.type, c.importance,
                       c.status, c.source, c.updated_at, c.embedding
                """,
                {"pid": project_id},
            )

            type_counts: dict[str, int] = {}
            quota = max(3, memory_limit // 3)
            scored = []

            while res.has_next():
                row = res.get_next()
                emb = row[8]
                if not emb:
                    continue
                sim = _dot(query_emb, emb)
                if sim < 0.3:
                    continue

                chunk_type = row[3]
                if type_counts.get(chunk_type, 0) >= quota:
                    continue

                importance  = int(row[4] or 3)
                updated_at  = row[7] or ""
                freshness   = _freshness_score(updated_at, freshness_decay_days)
                base_score  = sim * 0.8 + freshness * 0.2
                final_score = base_score * 0.85 + (importance / 5) * 0.15

                stale_warning = False
                try:
                    then = datetime.fromisoformat(updated_at)
                    if then.tzinfo is None:
                        then = then.replace(tzinfo=timezone.utc)
                    stale_warning = (datetime.now(timezone.utc) - then).days > self.stale_threshold_days
                except Exception:
                    pass

                scored.append({
                    "id":           row[0],
                    "title":        row[1],
                    "content":      row[2],
                    "type":         chunk_type,
                    "source":       row[6] or None,
                    "importance":   importance,
                    "status":       row[5],
                    "age":          _age_str(updated_at),
                    "stale_warning": stale_warning,
                    "similarity":   round(sim, 4),
                    "final_score":  round(final_score, 4),
                    "conflicts":    self._get_conflicts(row[0]),
                })
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

            scored.sort(key=lambda x: x["final_score"], reverse=True)
            results.extend(scored[:memory_limit])

        if not type_filter or "spec" in type_filter:
            res = self._conn.execute(
                """
                MATCH (c:Chunk)
                WHERE c.project_id = $pid AND c.type = 'spec' AND c.deleted_at = ''
                RETURN c.id, c.title, c.content, c.source, c.importance,
                       c.status, c.updated_at, c.embedding
                """,
                {"pid": project_id},
            )
            spec_scored = []
            while res.has_next():
                row = res.get_next()
                emb = row[7]
                if not emb:
                    continue
                sim = _dot(query_emb, emb)
                if sim < 0.3:
                    continue
                spec_scored.append({
                    "id":           row[0],
                    "title":        row[1],
                    "content":      row[2],
                    "type":         "spec",
                    "source":       row[3] or None,
                    "importance":   int(row[4] or 3),
                    "status":       row[5],
                    "age":          _age_str(row[6] or ""),
                    "stale_warning": False,
                    "similarity":   round(sim, 4),
                    "final_score":  round(sim, 4),
                    "conflicts":    [],
                })
            spec_scored.sort(key=lambda x: x["final_score"], reverse=True)
            results.extend(spec_scored[:spec_limit])

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    def _get_conflicts(self, chunk_id: str) -> list[dict]:
        try:
            res = self._conn.execute(
                """
                MATCH (a:Chunk {id: $id})-[r:CONTRADICTS]-(b:Chunk)
                WHERE b.status = 'active'
                RETURN b.id, b.title, r.confidence
                """,
                {"id": chunk_id},
            )
            out = []
            while res.has_next():
                row = res.get_next()
                out.append({"id": row[0], "title": row[1], "confidence": row[2]})
            return out
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Edge queries (for CLI + get_related_chunks tool)
    # ------------------------------------------------------------------

    def get_edges(
        self,
        chunk_id: str,
        edge_type: Optional[str] = None,
        depth: int = 1,
    ) -> list[dict]:
        types = [edge_type] if edge_type else list(EDGE_TYPES)
        edges = []
        for et in types:
            try:
                # outgoing
                res = self._conn.execute(
                    f"""
                    MATCH (a:Chunk {{id: $id}})-[r:{et}]->(b:Chunk)
                    RETURN b.id, b.title, b.type, b.status, r.confidence, r.created_by
                    """,
                    {"id": chunk_id},
                )
                while res.has_next():
                    row = res.get_next()
                    edges.append({
                        "chunk_id":   row[0],
                        "title":      row[1],
                        "type":       row[2],
                        "status":     row[3],
                        "confidence": row[4],
                        "created_by": row[5],
                        "edge_type":  et,
                        "direction":  "outgoing",
                    })
                # incoming
                res = self._conn.execute(
                    f"""
                    MATCH (b:Chunk)-[r:{et}]->(a:Chunk {{id: $id}})
                    RETURN b.id, b.title, b.type, b.status, r.confidence, r.created_by
                    """,
                    {"id": chunk_id},
                )
                while res.has_next():
                    row = res.get_next()
                    edges.append({
                        "chunk_id":   row[0],
                        "title":      row[1],
                        "type":       row[2],
                        "status":     row[3],
                        "confidence": row[4],
                        "created_by": row[5],
                        "edge_type":  et,
                        "direction":  "incoming",
                    })
            except Exception:
                pass

        if depth > 1:
            seen = {chunk_id} | {e["chunk_id"] for e in edges}
            second_hop = []
            for e in list(edges):
                for et in types:
                    try:
                        res = self._conn.execute(
                            f"""
                            MATCH (a:Chunk {{id: $id}})-[r:{et}]->(b:Chunk)
                            WHERE b.id <> $origin
                            RETURN b.id, b.title, b.type, b.status, r.confidence, r.created_by
                            """,
                            {"id": e["chunk_id"], "origin": chunk_id},
                        )
                        while res.has_next():
                            row = res.get_next()
                            if row[0] not in seen:
                                seen.add(row[0])
                                second_hop.append({
                                    "chunk_id":   row[0],
                                    "title":      row[1],
                                    "type":       row[2],
                                    "status":     row[3],
                                    "confidence": row[4],
                                    "created_by": row[5],
                                    "edge_type":  et,
                                    "direction":  "outgoing",
                                    "via":        e["chunk_id"],
                                })
                    except Exception:
                        pass
            edges.extend(second_hop)

        return edges

    def get_all_edges(self, project_id: str) -> list[dict]:
        edges = []
        for et in EDGE_TYPES:
            try:
                res = self._conn.execute(
                    f"""
                    MATCH (a:Chunk)-[r:{et}]->(b:Chunk)
                    WHERE a.project_id = $pid
                    RETURN a.id, a.title, a.type, b.id, b.title, b.type,
                           r.confidence, r.created_by
                    """,
                    {"pid": project_id},
                )
                while res.has_next():
                    row = res.get_next()
                    edges.append({
                        "from_id":    row[0],
                        "from_title": row[1],
                        "from_type":  row[2],
                        "to_id":      row[3],
                        "to_title":   row[4],
                        "to_type":    row[5],
                        "edge_type":  et,
                        "confidence": row[6],
                        "created_by": row[7],
                    })
            except Exception:
                pass
        return edges

    # ------------------------------------------------------------------
    # Spec indexing
    # ------------------------------------------------------------------

    def index_spec_file(self, project_id: str, file_path: str, chunks: list[dict]) -> int:
        # Compute IDs for incoming chunks
        incoming_ids: set[str] = set()
        for c in chunks:
            incoming_ids.add(hashlib.sha256(
                f"{project_id}spec{file_path}{normalize_title(c['title'])}".encode()
            ).hexdigest())

        # Find existing spec chunk IDs for this file
        try:
            res = self._conn.execute(
                """
                MATCH (c:Chunk)
                WHERE c.project_id = $pid AND c.type = 'spec' AND c.source STARTS WITH $file
                RETURN c.id
                """,
                {"pid": project_id, "file": file_path},
            )
            existing_ids: set[str] = set()
            while res.has_next():
                existing_ids.add(res.get_next()[0])
        except Exception:
            existing_ids = set()

        # DETACH DELETE only sections removed from source — preserves edges on surviving chunks
        for cid in existing_ids - incoming_ids:
            self._conn.execute("MATCH (c:Chunk {id: $id}) DETACH DELETE c", {"id": cid})

        now = datetime.now(timezone.utc).isoformat()
        count = 0
        for c in chunks:
            chunk_id = hashlib.sha256(
                f"{project_id}spec{file_path}{normalize_title(c['title'])}".encode()
            ).hexdigest()
            embedding = _embed(make_embedding_target("spec", c["title"], c.get("content", "")))
            if chunk_id in existing_ids:
                # Update content + embedding, preserve edges
                self._conn.execute(
                    """
                    MATCH (c:Chunk {id: $id})
                    SET c.content = $content, c.embedding = $embedding, c.updated_at = $now
                    """,
                    {"id": chunk_id, "content": c.get("content", ""), "embedding": embedding, "now": now},
                )
            else:
                self._conn.execute(
                    """
                    CREATE (:Chunk {
                        id: $id, project_id: $project_id, title: $title,
                        content: $content, type: $type, importance: $importance,
                        status: $status, source: $source, embedding: $embedding,
                        created_at: $created_at, updated_at: $updated_at, deleted_at: $deleted_at
                    })
                    """,
                    {
                        "id":         chunk_id,
                        "project_id": project_id,
                        "title":      c["title"],
                        "content":    c.get("content", ""),
                        "type":       "spec",
                        "importance": int(c.get("importance", 3)),
                        "status":     "active",
                        "source":     c.get("source", ""),
                        "embedding":  embedding,
                        "created_at": now,
                        "updated_at": now,
                        "deleted_at": "",
                    },
                )
            count += 1
        return count

    def clear_spec_chunks(self, project_id: str) -> None:
        try:
            res = self._conn.execute(
                "MATCH (c:Chunk) WHERE c.project_id = $pid AND c.type = 'spec' RETURN c.id",
                {"pid": project_id},
            )
            ids = []
            while res.has_next():
                ids.append(res.get_next()[0])
            for cid in ids:
                self._conn.execute("MATCH (c:Chunk {id: $id}) DETACH DELETE c", {"id": cid})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_all_memory_chunks(self, project_id: str, include_deprecated: bool = False) -> list[Chunk]:
        type_list = "', '".join(MEMORY_TYPES)
        status_clause = "" if include_deprecated else "AND c.status = 'active'"
        res = self._conn.execute(
            f"""
            MATCH (c:Chunk)
            WHERE c.project_id = $pid
              AND c.type IN ['{type_list}']
              AND c.deleted_at = ''
              {status_clause}
            RETURN {self._CHUNK_RETURN}
            """,
            {"pid": project_id},
        )
        chunks = []
        while res.has_next():
            chunks.append(self._row_to_chunk(res.get_next()))
        return chunks

    def get_spec_chunks(self, project_id: str) -> list[Chunk]:
        res = self._conn.execute(
            f"""
            MATCH (c:Chunk)
            WHERE c.project_id = $pid AND c.type = 'spec' AND c.deleted_at = ''
            RETURN {self._CHUNK_RETURN}
            """,
            {"pid": project_id},
        )
        chunks = []
        while res.has_next():
            chunks.append(self._row_to_chunk(res.get_next()))
        return chunks

    def reembed_all(self, project_id: str) -> int:
        chunks = self.get_all_memory_chunks(project_id, include_deprecated=True)
        for chunk in chunks:
            emb = _embed(make_embedding_target(chunk.type, chunk.title, chunk.content))
            self._conn.execute(
                "MATCH (c:Chunk {id: $id}) SET c.embedding = $emb",
                {"id": chunk.id, "emb": emb},
            )
        return len(chunks)

    def soft_delete(self, chunk_id: str) -> bool:
        full_id = self._expand_id(chunk_id)
        if not full_id:
            return False
        self._conn.execute("MATCH (c:Chunk {id: $id}) DETACH DELETE c", {"id": full_id})
        return True

    def _expand_id(self, chunk_id: str) -> Optional[str]:
        if len(chunk_id) == 64 and self._exists(chunk_id):
            return chunk_id
        res = self._conn.execute("MATCH (c:Chunk) RETURN c.id", {})
        while res.has_next():
            full_id = res.get_next()[0]
            if full_id.startswith(chunk_id):
                return full_id
        return None
