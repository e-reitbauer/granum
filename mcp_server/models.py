from __future__ import annotations

import hashlib
import re
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

ChunkType = Literal["decision", "preference", "file_state", "constraint", "spec"]
ChunkStatus = Literal["active", "deprecated"]

MEMORY_TYPES: tuple[ChunkType, ...] = ("decision", "preference", "file_state", "constraint")


def normalize_title(title: str) -> str:
    title = title.lower()
    title = title.translate(str.maketrans("", "", string.punctuation))
    title = re.sub(r"\s+", " ", title).strip()
    return title


def make_chunk_id(project_id: str, chunk_type: ChunkType, title: str) -> str:
    normalized = normalize_title(title)
    raw = f"{project_id}{chunk_type}{normalized}"
    return hashlib.sha256(raw.encode()).hexdigest()


def make_embedding_target(chunk_type: ChunkType, title: str, content: str = "") -> str:
    base = f"{chunk_type}: {normalize_title(title)}"
    if content:
        return f"{base}\n\n{content[:1000]}"
    return base


@dataclass
class Chunk:
    id: str
    project_id: str
    title: str
    content: str
    type: ChunkType
    importance: int = 3
    status: ChunkStatus = "active"
    source: Optional[str] = None
    created_at: str = field(default_factory=lambda: _now())
    updated_at: str = field(default_factory=lambda: _now())
    deleted_at: Optional[str] = None

    @classmethod
    def create(
        cls,
        project_id: str,
        title: str,
        content: str,
        chunk_type: ChunkType,
        importance: int = 3,
        source: Optional[str] = None,
    ) -> "Chunk":
        chunk_id = make_chunk_id(project_id, chunk_type, title)
        return cls(
            id=chunk_id,
            project_id=project_id,
            title=title,
            content=content,
            type=chunk_type,
            importance=importance,
            source=source,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "title": self.title,
            "content": self.content,
            "type": self.type,
            "importance": self.importance,
            "status": self.status,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deleted_at": self.deleted_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            id=d["id"],
            project_id=d["project_id"],
            title=d["title"],
            content=d["content"],
            type=d["type"],
            importance=d.get("importance", 3),
            status=d.get("status", "active"),
            source=d.get("source"),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
            deleted_at=d.get("deleted_at"),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
