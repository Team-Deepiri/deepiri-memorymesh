from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .models import AgentState, CompressedRecord, MemoryRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    project TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_messages_dedupe
ON memory_messages (project, provider, conversation_id, role, content, timestamp);

CREATE TABLE IF NOT EXISTS memory_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    method TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    embedding_json TEXT NOT NULL,
    FOREIGN KEY(message_id) REFERENCES memory_messages(id)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_embeddings_message
ON memory_embeddings (message_id);

CREATE TABLE IF NOT EXISTS agent_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project TEXT NOT NULL,
    agent TEXT NOT NULL,
    state_key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project, agent, state_key)
);
"""


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def insert_messages(self, records: Iterable[MemoryRecord]) -> int:
        rows = [
            (
                r.provider,
                r.project,
                r.conversation_id,
                r.role,
                r.content,
                r.timestamp,
                r.metadata_json,
            )
            for r in records
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            cur = conn.executemany(
                """
                INSERT OR IGNORE INTO memory_messages
                (provider, project, conversation_id, role, content, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            return cur.rowcount

    def list_messages(self, project: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT id, provider, project, conversation_id, role, content, timestamp, metadata_json
                FROM memory_messages
                WHERE project = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (project,),
            )
            return list(cur.fetchall())

    def upsert_summary(self, rec: CompressedRecord) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_summaries (project, conversation_id, summary, method, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (rec.project, rec.conversation_id, rec.summary, rec.method, rec.created_at),
            )
            conn.commit()

    def set_agent_state(self, rec: AgentState) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_state (project, agent, state_key, value, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(project, agent, state_key)
                DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (rec.project, rec.agent, rec.key, rec.value, rec.updated_at),
            )
            conn.commit()

    def get_agent_state(self, project: str, agent: str, key: str) -> str | None:
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT value FROM agent_state
                WHERE project = ? AND agent = ? AND state_key = ?
                """,
                (project, agent, key),
            )
            row = cur.fetchone()
            return None if row is None else str(row["value"])

    def save_embedding(self, message_id: int, embedding_json: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_embeddings (message_id, embedding_json)
                VALUES (?, ?)
                ON CONFLICT(message_id) DO UPDATE SET embedding_json = excluded.embedding_json
                """,
                (message_id, embedding_json),
            )
            conn.commit()

    def list_summaries(self, project: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT conversation_id, summary, method, created_at
                FROM memory_summaries
                WHERE project = ?
                ORDER BY created_at DESC
                """,
                (project,),
            )
            return list(cur.fetchall())

    def list_embeddings(self, project: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT m.id AS message_id, m.content, m.provider, m.conversation_id, e.embedding_json
                FROM memory_embeddings e
                JOIN memory_messages m ON e.message_id = m.id
                WHERE m.project = ?
                """,
                (project,),
            )
            return list(cur.fetchall())
