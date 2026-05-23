from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import logging
import warnings

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Suppress HuggingFace download noise and auth warnings
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

from .models import (
    MEMORY_TYPES,
    Chunk,
    ChunkStatus,
    ChunkType,
    make_chunk_id,
    make_embedding_target,
    normalize_title,
)

MAX_CONTENT_CHARS = 2000
TRUNCATION_SUFFIX = "...[truncated — full content exceeded 500 token limit]"

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        with open(os.devnull, "w") as devnull:
            import sys
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                _model = SentenceTransformer("all-MiniLM-L6-v2")
            finally:
                sys.stderr = old_stderr
    return _model


def _embed(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()


def _truncate_content(content: str) -> str:
    if len(content) <= MAX_CONTENT_CHARS:
        return content
    cutoff = MAX_CONTENT_CHARS - len(TRUNCATION_SUFFIX)
    return content[:cutoff] + TRUNCATION_SUFFIX


def _age_str(updated_at: str) -> str:
    try:
        then = datetime.fromisoformat(updated_at)
        delta = datetime.now(timezone.utc) - then
        days = delta.days
        if days == 0:
            hours = delta.seconds // 3600
            return f"{hours}h ago" if hours > 0 else "just now"
        if days == 1:
            return "1 day ago"
        return f"{days} days ago"
    except Exception:
        return "unknown"


def _freshness_score(updated_at: str, decay_days: int = 90) -> float:
    try:
        then = datetime.fromisoformat(updated_at)
        delta = datetime.now(timezone.utc) - then
        age_days = delta.total_seconds() / 86400
        return max(0.0, 1.0 - age_days / decay_days)
    except Exception:
        return 0.0


class GranumDB:
    def __init__(self, db_path: Path, ndjson_path: Path, stale_threshold_days: int = 7):
        self.db_path = db_path
        self.ndjson_path = ndjson_path
        self.stale_threshold_days = stale_threshold_days

        self._client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="granum",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Import / export
    # ------------------------------------------------------------------

    def import_ndjson(self) -> int:
        if not self.ndjson_path.exists():
            return 0
        count = 0
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
                    # Tombstone: remove from index if present
                    try:
                        self._collection.delete(ids=[chunk.id])
                    except Exception:
                        pass
                    continue
                self._upsert_to_chroma(chunk)
                count += 1
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

        # Fetch all memory chunks for project (spec chunks excluded)
        results = self._collection.get(
            where={"$and": [{"project_id": project_id}, {"type": {"$in": list(MEMORY_TYPES)}}]},
            include=["metadatas", "documents"],
        )
        for i, chunk_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            d = {
                "id": chunk_id,
                "project_id": meta["project_id"],
                "title": meta["title"],
                "content": results["documents"][i],
                "type": meta["type"],
                "importance": meta["importance"],
                "status": meta["status"],
                "source": meta.get("source"),
                "created_at": meta["created_at"],
                "updated_at": meta["updated_at"],
                "deleted_at": meta.get("deleted_at"),
            }
            existing[chunk_id] = d

        self.ndjson_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ndjson_path.open("w") as f:
            for d in existing.values():
                f.write(json.dumps(d) + "\n")

    # ------------------------------------------------------------------
    # Core upsert
    # ------------------------------------------------------------------

    def _upsert_to_chroma(self, chunk: Chunk) -> None:
        embedding = _embed(make_embedding_target(chunk.type, chunk.title, chunk.content))
        metadata = {
            "project_id": chunk.project_id,
            "title": chunk.title,
            "type": chunk.type,
            "importance": chunk.importance,
            "status": chunk.status,
            "created_at": chunk.created_at,
            "updated_at": chunk.updated_at,
        }
        if chunk.source:
            metadata["source"] = chunk.source
        if chunk.deleted_at:
            metadata["deleted_at"] = chunk.deleted_at

        self._collection.upsert(
            ids=[chunk.id],
            embeddings=[embedding],
            documents=[chunk.content],
            metadatas=[metadata],
        )

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

        existing = self._get_by_id(chunk_id)
        now = datetime.now(timezone.utc).isoformat()

        if existing:
            existing.content = content
            existing.updated_at = now
            existing.importance = importance
            self._upsert_to_chroma(existing)
            action = "updated"
        else:
            chunk = Chunk.create(
                project_id=project_id,
                title=title,
                content=content,
                chunk_type=chunk_type,
                importance=importance,
            )
            self._upsert_to_chroma(chunk)
            action = "created"

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
                chunk = self._get_by_id(cid)
                if chunk:
                    chunk.deleted_at = now
                    self._upsert_to_chroma(chunk)
                    self._collection.delete(ids=[cid])
            return {"action": "deleted", "ids": chunk_ids}

        elif action == "deprecate":
            for cid in chunk_ids:
                chunk = self._get_by_id(cid)
                if chunk:
                    chunk.status = "deprecated"
                    chunk.updated_at = now
                    self._upsert_to_chroma(chunk)
            return {"action": "deprecated", "ids": chunk_ids}

        elif action == "merge":
            if not all([merged_title, merged_content, merged_type, project_id]):
                raise ValueError("merge requires merged_title, merged_content, merged_type, project_id")
            for cid in chunk_ids:
                chunk = self._get_by_id(cid)
                if chunk:
                    chunk.status = "deprecated"
                    chunk.updated_at = now
                    self._upsert_to_chroma(chunk)
            result = self.save_context(
                project_id=project_id,
                title=merged_title,
                content=merged_content,
                chunk_type=merged_type,
                importance=merged_importance or 3,
            )
            return {"action": "merged", "deprecated_ids": chunk_ids, "new_id": result["id"]}

        elif action == "update":
            if len(chunk_ids) != 1:
                raise ValueError("update requires exactly one chunk_id")
            chunk = self._get_by_id(chunk_ids[0])
            if not chunk:
                raise ValueError(f"chunk not found: {chunk_ids[0]}")
            if merged_content:
                chunk.content = _truncate_content(merged_content)
            if merged_importance:
                chunk.importance = merged_importance
            chunk.updated_at = now
            self._upsert_to_chroma(chunk)
            return {"action": "updated", "id": chunk_ids[0]}

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
        query_embedding = _embed(query)
        results = []

        # Memory query
        memory_types = list(type_filter) if type_filter else list(MEMORY_TYPES)
        memory_types = [t for t in memory_types if t != "spec"]
        if memory_types:
            memory_results = self._query_chroma(
                project_id=project_id,
                query_embedding=query_embedding,
                type_filter=memory_types,
                status_filter="active",
                n_results=memory_limit * 3,  # over-fetch for per-type quota
            )
            memory_chunks = self._score_memory(
                memory_results, freshness_decay_days, limit=memory_limit
            )
            results.extend(memory_chunks)

        # Spec query (only if not filtered to memory-only)
        if not type_filter or "spec" in type_filter:
            spec_results = self._query_chroma(
                project_id=project_id,
                query_embedding=query_embedding,
                type_filter=["spec"],
                status_filter=None,
                n_results=spec_limit,
            )
            spec_chunks = self._score_spec(spec_results, limit=spec_limit)
            results.extend(spec_chunks)

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    def _query_chroma(
        self,
        project_id: str,
        query_embedding: list[float],
        type_filter: list[str],
        status_filter: Optional[str],
        n_results: int,
    ) -> dict:
        where: dict = {"$and": [
            {"project_id": project_id},
            {"type": {"$in": type_filter}},
        ]}
        if status_filter:
            where["$and"].append({"status": status_filter})

        try:
            return self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            return {"ids": [[]], "metadatas": [[]], "documents": [[]], "distances": [[]]}

    def _score_memory(self, results: dict, decay_days: int, limit: int) -> list[dict]:
        ids = results["ids"][0]
        metas = results["metadatas"][0]
        docs = results["documents"][0]
        dists = results["distances"][0]

        # per-type quota tracking
        type_counts: dict[str, int] = {}
        scored = []

        for i, chunk_id in enumerate(ids):
            meta = metas[i]
            chunk_type = meta["type"]
            if type_counts.get(chunk_type, 0) >= max(3, limit // 3):
                continue

            similarity = max(0.0, 1.0 - dists[i])
            freshness = _freshness_score(meta.get("updated_at", ""), decay_days)
            importance = meta.get("importance", 3)
            base_score = similarity * 0.8 + freshness * 0.2
            final_score = base_score * (importance / 5)

            stale_warning = False
            try:
                updated = datetime.fromisoformat(meta.get("updated_at", ""))
                age_days = (datetime.now(timezone.utc) - updated).days
                stale_warning = age_days > self.stale_threshold_days
            except Exception:
                pass

            scored.append({
                "id": chunk_id,
                "title": meta["title"],
                "content": docs[i],
                "type": chunk_type,
                "source": meta.get("source"),
                "importance": importance,
                "status": meta.get("status", "active"),
                "age": _age_str(meta.get("updated_at", "")),
                "stale_warning": stale_warning,
                "similarity": round(similarity, 4),
                "final_score": round(final_score, 4),
            })
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:limit]

    def _score_spec(self, results: dict, limit: int) -> list[dict]:
        ids = results["ids"][0]
        metas = results["metadatas"][0]
        docs = results["documents"][0]
        dists = results["distances"][0]

        scored = []
        for i, chunk_id in enumerate(ids):
            meta = metas[i]
            similarity = max(0.0, 1.0 - dists[i])
            scored.append({
                "id": chunk_id,
                "title": meta["title"],
                "content": docs[i],
                "type": "spec",
                "source": meta.get("source"),
                "importance": meta.get("importance", 3),
                "status": meta.get("status", "active"),
                "age": _age_str(meta.get("updated_at", "")),
                "stale_warning": False,
                "similarity": round(similarity, 4),
                "final_score": round(similarity, 4),
            })

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Spec indexing
    # ------------------------------------------------------------------

    def index_spec_file(self, project_id: str, file_path: str, chunks: list[dict]) -> int:
        # Clear existing spec chunks for this file
        try:
            existing = self._collection.get(
                where={"$and": [
                    {"project_id": project_id},
                    {"type": "spec"},
                    {"source_file": file_path},
                ]},
                include=[],
            )
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
        except Exception:
            pass

        count = 0
        for c in chunks:
            chunk = Chunk.create(
                project_id=project_id,
                title=c["title"],
                content=c["content"],
                chunk_type="spec",
                importance=c.get("importance", 3),
                source=c.get("source"),
            )
            # Override ID to include source_file — prevents title collisions across spec files
            chunk.id = hashlib.sha256(
                f"{project_id}spec{file_path}{normalize_title(c['title'])}".encode()
            ).hexdigest()
            # Store source_file separately for deletion lookup
            embedding = _embed(make_embedding_target(chunk.type, chunk.title, chunk.content))
            self._collection.upsert(
                ids=[chunk.id],
                embeddings=[embedding],
                documents=[chunk.content],
                metadatas=[{
                    "project_id": chunk.project_id,
                    "title": chunk.title,
                    "type": chunk.type,
                    "importance": chunk.importance,
                    "status": chunk.status,
                    "created_at": chunk.created_at,
                    "updated_at": chunk.updated_at,
                    "source": chunk.source or "",
                    "source_file": file_path,
                }],
            )
            count += 1
        return count

    def clear_spec_chunks(self, project_id: str) -> None:
        try:
            existing = self._collection.get(
                where={"$and": [
                    {"project_id": project_id},
                    {"type": "spec"},
                ]},
                include=[],
            )
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        try:
            result = self._collection.get(ids=[chunk_id], include=["metadatas", "documents"])
            if not result["ids"]:
                return None
            meta = result["metadatas"][0]
            return Chunk(
                id=chunk_id,
                project_id=meta["project_id"],
                title=meta["title"],
                content=result["documents"][0],
                type=meta["type"],
                importance=meta.get("importance", 3),
                status=meta.get("status", "active"),
                source=meta.get("source"),
                created_at=meta.get("created_at", ""),
                updated_at=meta.get("updated_at", ""),
                deleted_at=meta.get("deleted_at"),
            )
        except Exception:
            return None

    def get_all_memory_chunks(self, project_id: str, include_deprecated: bool = False) -> list[Chunk]:
        where: dict = {"$and": [
            {"project_id": project_id},
            {"type": {"$in": list(MEMORY_TYPES)}},
        ]}
        if not include_deprecated:
            where["$and"].append({"status": "active"})

        try:
            result = self._collection.get(where=where, include=["metadatas", "documents"])
        except Exception:
            return []

        chunks = []
        for i, chunk_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            chunks.append(Chunk(
                id=chunk_id,
                project_id=meta["project_id"],
                title=meta["title"],
                content=result["documents"][i],
                type=meta["type"],
                importance=meta.get("importance", 3),
                status=meta.get("status", "active"),
                source=meta.get("source"),
                created_at=meta.get("created_at", ""),
                updated_at=meta.get("updated_at", ""),
                deleted_at=meta.get("deleted_at"),
            ))
        return chunks

    def get_spec_chunks(self, project_id: str) -> list[Chunk]:
        where: dict = {"$and": [
            {"project_id": project_id},
            {"type": "spec"},
        ]}
        try:
            result = self._collection.get(where=where, include=["metadatas", "documents"])
        except Exception:
            return []

        chunks = []
        for i, chunk_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            chunks.append(Chunk(
                id=chunk_id,
                project_id=meta["project_id"],
                title=meta["title"],
                content=result["documents"][i],
                type="spec",
                importance=meta.get("importance", 3),
                status=meta.get("status", "active"),
                source=meta.get("source"),
                created_at=meta.get("created_at", ""),
                updated_at=meta.get("updated_at", ""),
                deleted_at=meta.get("deleted_at"),
            ))
        return chunks

    def reembed_all(self, project_id: str) -> int:
        """Re-embed all memory chunks with current embedding strategy."""
        chunks = self.get_all_memory_chunks(project_id, include_deprecated=True)
        for chunk in chunks:
            self._upsert_to_chroma(chunk)
        return len(chunks)

    def soft_delete(self, chunk_id: str) -> bool:
        chunk = self._get_by_id(chunk_id)
        if not chunk:
            return False
        now = datetime.now(timezone.utc).isoformat()
        chunk.deleted_at = now
        self._upsert_to_chroma(chunk)
        self._collection.delete(ids=[chunk_id])
        return True
