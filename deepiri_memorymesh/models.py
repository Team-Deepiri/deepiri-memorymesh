from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MemoryRecord:
    provider: str
    project: str
    conversation_id: str
    role: str
    content: str
    timestamp: str = field(default_factory=now_iso)
    metadata_json: str = "{}"
    # Optional stable provider turn identity (T16/T20). NULL rows keep the
    # content/timestamp unique index; non-NULL rows also use the partial unique
    # index on (project, provider, conversation_id, source_key).
    source_key: str | None = None


@dataclass(slots=True)
class CompressedRecord:
    project: str
    conversation_id: str
    summary: str
    method: str
    created_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class AgentState:
    project: str
    agent: str
    key: str
    value: str
    updated_at: str = field(default_factory=now_iso)
