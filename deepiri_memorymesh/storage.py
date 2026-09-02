"""Canonical Memory Mesh platform SQLite store."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Iterable, Literal

from .migrations import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_BUSY_TIMEOUT_MS,
    ensure_migrated,
    current_schema_version,
)
from .models import AgentState, CompressedRecord, MemoryRecord
from .search_index import index_message_terms

BUSY_TIMEOUT_MS = DEFAULT_BUSY_TIMEOUT_MS
BUSY_TIMEOUT_SECONDS = BUSY_TIMEOUT_MS / 1000.0

# Bounded busy/locked retries for write helpers (T13).
BUSY_RETRY_ATTEMPTS = 8
BUSY_RETRY_BASE_DELAY_S = 0.02

SchemaKind = Literal["empty", "platform", "legacy", "unknown", "corrupt"]


def sqlite_readonly_uri(path: Path) -> str:
    """Build a read-only SQLite URI with correctly encoded path characters."""
    return f"{path.expanduser().resolve().as_uri()}?mode=ro"


def _table_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return {str(row[0]) for row in cur.fetchall()}


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {str(row[1]) for row in cur.fetchall()}


def _is_exact_legacy_memories_table(conn: sqlite3.Connection) -> bool:
    """Historical simple schema: ``memories(content, embedding, created_at)``."""
    cols = _column_names(conn, "memories")
    required = {"content", "embedding", "created_at"}
    return required.issubset(cols)


def detect_db_schema(db_path: Path) -> SchemaKind:
    """Classify a SQLite path without modifying it.

    - empty: missing file, zero-byte file, or no user tables
    - platform: canonical ``memory_messages`` present and no legacy ``memories``
    - legacy: exact historical ``memories`` table only (no platform tables)
    - unknown: unrelated, mixed, or ambiguous schemas
    - corrupt: SQLite cannot open/read the file
    """
    if not db_path.exists():
        return "empty"
    try:
        if db_path.stat().st_size == 0:
            return "empty"
    except OSError:
        return "corrupt"

    try:
        # Read-only URI so detection never writes journals/WAL against the path.
        conn = sqlite3.connect(sqlite_readonly_uri(db_path), uri=True)
    except sqlite3.Error:
        return "corrupt"
    try:
        try:
            tables = _table_names(conn)
        except sqlite3.Error:
            return "corrupt"
        if not tables:
            return "empty"

        has_platform = "memory_messages" in tables
        has_memories = "memories" in tables
        if has_platform and has_memories:
            return "unknown"  # ambiguous mixed schema
        if has_platform:
            return "platform"
        if has_memories:
            # Fail closed unless this is the exact historical simple table and
            # there are no other memory_* platform tables.
            if any(t.startswith("memory_") for t in tables):
                return "unknown"
            try:
                if not _is_exact_legacy_memories_table(conn):
                    return "unknown"
            except sqlite3.Error:
                return "corrupt"
            # Allow sqlite_sequence / only memories (+ optional id PK internals).
            extras = tables - {"memories", "sqlite_sequence"}
            if extras:
                return "unknown"
            return "legacy"
        return "unknown"
    finally:
        conn.close()


class LegacySchemaError(RuntimeError):
    """Raised when a caller points Memory at a legacy simple-memory database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        super().__init__(
            f"Database at {db_path} uses the legacy simple-memory schema "
            f"(table 'memories'), not the canonical Memory Mesh platform schema. "
            f"It will not be rewritten automatically. Import explicitly with: "
            f"memorymesh import-legacy-memory --source {db_path}"
        )


class AmbiguousSchemaError(RuntimeError):
    """Raised for corrupt/unrelated/mixed schemas that must not be written."""

    def __init__(self, db_path: Path, kind: str):
        self.db_path = db_path
        self.kind = kind
        super().__init__(
            f"Database at {db_path} has schema kind {kind!r} and will not be "
            "initialized or overwritten. Use a new path or "
            "memorymesh import-legacy-memory when importing a legacy source."
        )


@dataclass(slots=True)
class DatabaseStatus:
    """Small programmatic diagnostics for the platform database (T13)."""

    db_path: Path
    schema_version: int
    latest_schema_version: int
    journal_mode: str
    foreign_keys: bool
    busy_timeout_ms: int
    wal_active: bool


def _is_busy_error(exc: BaseException) -> bool:
    if isinstance(exc, sqlite3.OperationalError):
        msg = str(exc).lower()
        return "locked" in msg or "busy" in msg
    return False


def with_busy_retry(fn, *, attempts: int = BUSY_RETRY_ATTEMPTS):
    """Retry *fn* on SQLite busy/locked only, with bounded backoff."""
    delay = BUSY_RETRY_BASE_DELAY_S
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not _is_busy_error(exc) or attempt + 1 >= attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.5)
    assert last is not None
    raise last


class MemoryStore:
    def __init__(
        self,
        db_path: Path,
        *,
        busy_timeout_ms: int = BUSY_TIMEOUT_MS,
        key_file: Path | None = None,
    ):
        self.db_path = db_path
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.key_file = Path(key_file) if key_file is not None else None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Lazily resolved after migrations / encryption_meta is readable.
        self._crypto_ctx: object | None = None
        self._crypto_enabled: bool | None = None
        self._terms_mode: str = "plaintext"

    def connect(self) -> sqlite3.Connection:
        """Return a raw SQLite connection (caller must close it).

        Prefer :meth:`connection` or :meth:`transaction` for automatic cleanup.
        Does **not** change journal_mode (WAL is enabled during init/migrate).
        """
        timeout_s = max(self.busy_timeout_ms / 1000.0, 0.001)
        conn = sqlite3.connect(self.db_path, timeout=timeout_s)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection and always close it."""
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection; commit on success, rollback on error, always close."""
        with self.connection() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def init(self) -> None:
        kind = detect_db_schema(self.db_path)
        if kind == "legacy":
            raise LegacySchemaError(self.db_path)
        if kind in {"unknown", "corrupt"}:
            raise AmbiguousSchemaError(self.db_path, kind)
        ensure_migrated(self.db_path, busy_timeout_ms=self.busy_timeout_ms)
        # Refresh crypto state after schema is ready (no-op when plaintext).
        self._invalidate_crypto_cache()
        self.encryption_context()

    def _invalidate_crypto_cache(self) -> None:
        self._crypto_ctx = None
        self._crypto_enabled = None
        self._terms_mode = "plaintext"

    def encryption_enabled(self) -> bool:
        """True when ``encryption_meta.enabled`` is set for this database."""
        ctx = self.encryption_context()
        return ctx is not None

    def encryption_context(self):
        """Return an :class:`EncryptionContext` when encryption is enabled.

        Returns ``None`` for plaintext databases. When encryption is enabled,
        the master key is loaded from ``MEMORYMESH_ENCRYPTION_KEY`` or
        :attr:`key_file`. Key material is never logged.
        """
        if self._crypto_enabled is False:
            return None
        if self._crypto_enabled is True:
            return self._crypto_ctx

        if not self.db_path.exists() or self.db_path.stat().st_size == 0:
            self._crypto_enabled = False
            return None

        with self.connection() as conn:
            tables = _table_names(conn)
            if "encryption_meta" not in tables:
                self._crypto_enabled = False
                return None
            row = conn.execute(
                "SELECT enabled, key_id, db_identity, terms_mode "
                "FROM encryption_meta WHERE id = 1"
            ).fetchone()
            if row is None or not bool(row["enabled"]):
                self._crypto_enabled = False
                self._terms_mode = "plaintext" if row is None else str(row["terms_mode"] or "plaintext")
                return None
            db_identity = str(row["db_identity"])
            self._terms_mode = str(row["terms_mode"] or "keyed")
            expected_kid = row["key_id"]

        from .crypto import EncryptionContext, InvalidKeyMaterialError, load_master_key

        try:
            master = load_master_key(key_file=self.key_file)
        except InvalidKeyMaterialError as exc:
            raise InvalidKeyMaterialError(
                "Database encryption is enabled but no usable key was found. "
                "Set MEMORYMESH_ENCRYPTION_KEY or configure encryption_key_file. "
                f"({exc})"
            ) from exc
        ctx = EncryptionContext(master, db_identity)
        if expected_kid and ctx.key_id != expected_kid:
            from .crypto import WrongKeyError

            raise WrongKeyError(
                "Loaded encryption key does not match the database key id."
            )
        self._crypto_ctx = ctx
        self._crypto_enabled = True
        return ctx

    def _term_mapper(self):
        ctx = self.encryption_context()
        if ctx is None or self._terms_mode != "keyed":
            return None
        from .crypto import keyed_term_token

        return lambda term: keyed_term_token(ctx, term)

    def map_query_terms(self, plaintext_terms: list[str]) -> list[str]:
        """Map query tokens to stored index tokens (HMAC when keyed)."""
        mapper = self._term_mapper()
        if mapper is None:
            return list(plaintext_terms)
        return [mapper(t) for t in plaintext_terms]

    def _has_content_fingerprint_column(self, conn: sqlite3.Connection) -> bool:
        return "content_fingerprint" in _column_names(conn, "memory_messages")

    @staticmethod
    def _next_row_id(conn: sqlite3.Connection, table: str) -> int:
        """Allocate the next INTEGER PRIMARY KEY under an open write transaction."""
        row = conn.execute(f"SELECT COALESCE(MAX(id), 0) + 1 FROM {table}").fetchone()
        return int(row[0])

    def _encrypt_message_fields(
        self,
        message_id: int,
        *,
        project: str,
        provider: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata_json: str,
    ) -> tuple[str, str, str | None]:
        """Return ``(content, metadata_json, fingerprint)`` ready for SQL storage.

        When encryption is enabled, values are sealed *before* any INSERT/UPDATE so
        plaintext never lands in the database file — even transiently.
        """
        ctx = self.encryption_context()
        if ctx is None:
            return content, metadata_json, None
        from .crypto import content_fingerprint, encrypt_field

        ct_content = encrypt_field(
            ctx, content, table="memory_messages", column="content", row_identity=str(message_id)
        )
        ct_meta = encrypt_field(
            ctx,
            metadata_json,
            table="memory_messages",
            column="metadata_json",
            row_identity=str(message_id),
        )
        fp = content_fingerprint(
            ctx,
            project=project,
            provider=provider,
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        return ct_content, ct_meta, fp

    def _encrypt_field_value(
        self, plaintext: str, *, table: str, column: str, row_identity: int
    ) -> str:
        ctx = self.encryption_context()
        if ctx is None:
            return plaintext
        from .crypto import encrypt_field

        return encrypt_field(
            ctx, plaintext, table=table, column=column, row_identity=str(row_identity)
        )

    def _seal_embedding(self, conn: sqlite3.Connection, embedding_row_id: int, embedding_json: str) -> str:
        """Encrypt *embedding_json* for *embedding_row_id* and UPDATE the row."""
        sealed = self._encrypt_field_value(
            embedding_json,
            table="memory_embeddings",
            column="embedding_json",
            row_identity=embedding_row_id,
        )
        if self.encryption_context() is not None:
            conn.execute(
                "UPDATE memory_embeddings SET embedding_json = ? WHERE id = ?",
                (sealed, embedding_row_id),
            )
        return sealed

    def _seal_summary(self, conn: sqlite3.Connection, summary_id: int, summary: str) -> str:
        sealed = self._encrypt_field_value(
            summary, table="memory_summaries", column="summary", row_identity=summary_id
        )
        if self.encryption_context() is not None:
            conn.execute(
                "UPDATE memory_summaries SET summary = ? WHERE id = ?",
                (sealed, summary_id),
            )
        return sealed

    def _seal_agent_value(self, conn: sqlite3.Connection, state_id: int, value: str) -> str:
        sealed = self._encrypt_field_value(
            value, table="agent_state", column="value", row_identity=state_id
        )
        if self.encryption_context() is not None:
            conn.execute(
                "UPDATE agent_state SET value = ? WHERE id = ?",
                (sealed, state_id),
            )
        return sealed

    def _unseal_message_dict(self, row: sqlite3.Row | dict) -> dict:
        data = dict(row)
        ctx = self.encryption_context()
        if ctx is None:
            return data
        from .crypto import decrypt_field, is_envelope

        mid = str(data.get("id") or data.get("message_id") or "")
        content = data.get("content")
        if isinstance(content, str) and is_envelope(content) and mid:
            data["content"] = decrypt_field(
                ctx, content, table="memory_messages", column="content", row_identity=mid
            )
        meta = data.get("metadata_json")
        if isinstance(meta, str) and is_envelope(meta) and mid:
            data["metadata_json"] = decrypt_field(
                ctx, meta, table="memory_messages", column="metadata_json", row_identity=mid
            )
        return data

    def _unseal_embedding_dict(self, row: sqlite3.Row | dict) -> dict:
        data = dict(row)
        ctx = self.encryption_context()
        if ctx is None:
            return data
        from .crypto import decrypt_field, is_envelope

        emb = data.get("embedding_json")
        emb_id = data.get("embedding_id")
        if isinstance(emb, str) and is_envelope(emb) and emb_id is not None:
            data["embedding_json"] = decrypt_field(
                ctx,
                emb,
                table="memory_embeddings",
                column="embedding_json",
                row_identity=str(emb_id),
            )
        content = data.get("content")
        mid = data.get("message_id") or data.get("id")
        if isinstance(content, str) and is_envelope(content) and mid is not None:
            data["content"] = decrypt_field(
                ctx, content, table="memory_messages", column="content", row_identity=str(mid)
            )
        return data

    def _unseal_summary_dict(self, row: sqlite3.Row | dict) -> dict:
        data = dict(row)
        ctx = self.encryption_context()
        if ctx is None:
            return data
        from .crypto import decrypt_field, is_envelope

        summary = data.get("summary")
        sid = data.get("id")
        if isinstance(summary, str) and is_envelope(summary) and sid is not None:
            data["summary"] = decrypt_field(
                ctx, summary, table="memory_summaries", column="summary", row_identity=str(sid)
            )
        return data

    def database_status(self) -> DatabaseStatus:
        """Return schema/journal/FK/busy diagnostics for this store."""
        with self.connection() as conn:
            try:
                version = current_schema_version(conn)
            except Exception:
                version = 0
            journal = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            busy = conn.execute("PRAGMA busy_timeout").fetchone()
            busy_ms = int(busy[0]) if busy is not None else self.busy_timeout_ms
            return DatabaseStatus(
                db_path=self.db_path,
                schema_version=version,
                latest_schema_version=CURRENT_SCHEMA_VERSION,
                journal_mode=journal,
                foreign_keys=bool(fk),
                busy_timeout_ms=busy_ms,
                wal_active=(journal == "wal"),
            )

    def _message_has_source_key_column(self, conn: sqlite3.Connection) -> bool:
        return "source_key" in _column_names(conn, "memory_messages")

    def _terms_table_ready(self, conn: sqlite3.Connection) -> bool:
        return "memory_message_terms" in _table_names(conn)

    def insert_messages(self, records: Iterable[MemoryRecord]) -> int:
        rows = list(records)
        if not rows:
            return 0

        def _do() -> int:
            # Resolve crypto *before* BEGIN IMMEDIATE to avoid nested-connection lockup.
            ctx = self.encryption_context()
            term_mapper = self._term_mapper()
            encrypted = ctx is not None
            with self.connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    has_sk = self._message_has_source_key_column(conn)
                    terms_ready = self._terms_table_ready(conn)
                    inserted = 0
                    for r in rows:
                        if encrypted and not r.source_key:
                            # Without source_key, skip duplicates via fingerprint.
                            from .crypto import content_fingerprint

                            fp = content_fingerprint(
                                ctx,
                                project=r.project,
                                provider=r.provider,
                                conversation_id=r.conversation_id,
                                role=r.role,
                                content=r.content,
                            )
                            if self._has_content_fingerprint_column(conn):
                                exists = conn.execute(
                                    """
                                    SELECT id FROM memory_messages
                                    WHERE project = ? AND provider = ?
                                      AND conversation_id = ? AND role = ?
                                      AND content_fingerprint = ?
                                    LIMIT 1
                                    """,
                                    (
                                        r.project,
                                        r.provider,
                                        r.conversation_id,
                                        r.role,
                                        fp,
                                    ),
                                ).fetchone()
                                if exists is not None:
                                    continue

                        # Encrypt before insert when enabled (preallocate id for AAD).
                        if encrypted:
                            new_id = self._next_row_id(conn, "memory_messages")
                            store_content, store_meta, store_fp = self._encrypt_message_fields(
                                new_id,
                                project=r.project,
                                provider=r.provider,
                                conversation_id=r.conversation_id,
                                role=r.role,
                                content=r.content,
                                metadata_json=r.metadata_json,
                            )
                            if has_sk and self._has_content_fingerprint_column(conn):
                                cur = conn.execute(
                                    """
                                    INSERT OR IGNORE INTO memory_messages
                                    (id, provider, project, conversation_id, role, content,
                                     timestamp, metadata_json, source_key, content_fingerprint)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        new_id,
                                        r.provider,
                                        r.project,
                                        r.conversation_id,
                                        r.role,
                                        store_content,
                                        r.timestamp,
                                        store_meta,
                                        r.source_key,
                                        store_fp,
                                    ),
                                )
                            elif has_sk:
                                cur = conn.execute(
                                    """
                                    INSERT OR IGNORE INTO memory_messages
                                    (id, provider, project, conversation_id, role, content,
                                     timestamp, metadata_json, source_key)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        new_id,
                                        r.provider,
                                        r.project,
                                        r.conversation_id,
                                        r.role,
                                        store_content,
                                        r.timestamp,
                                        store_meta,
                                        r.source_key,
                                    ),
                                )
                            elif self._has_content_fingerprint_column(conn):
                                cur = conn.execute(
                                    """
                                    INSERT OR IGNORE INTO memory_messages
                                    (id, provider, project, conversation_id, role, content,
                                     timestamp, metadata_json, content_fingerprint)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        new_id,
                                        r.provider,
                                        r.project,
                                        r.conversation_id,
                                        r.role,
                                        store_content,
                                        r.timestamp,
                                        store_meta,
                                        store_fp,
                                    ),
                                )
                            else:
                                cur = conn.execute(
                                    """
                                    INSERT OR IGNORE INTO memory_messages
                                    (id, provider, project, conversation_id, role, content,
                                     timestamp, metadata_json)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        new_id,
                                        r.provider,
                                        r.project,
                                        r.conversation_id,
                                        r.role,
                                        store_content,
                                        r.timestamp,
                                        store_meta,
                                    ),
                                )
                            if not (cur.rowcount and cur.lastrowid):
                                continue
                        elif has_sk:
                            cur = conn.execute(
                                """
                                INSERT OR IGNORE INTO memory_messages
                                (provider, project, conversation_id, role, content,
                                 timestamp, metadata_json, source_key)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    r.provider,
                                    r.project,
                                    r.conversation_id,
                                    r.role,
                                    r.content,
                                    r.timestamp,
                                    r.metadata_json,
                                    r.source_key,
                                ),
                            )
                            if not (cur.rowcount and cur.lastrowid):
                                continue
                            new_id = int(cur.lastrowid)
                        else:
                            cur = conn.execute(
                                """
                                INSERT OR IGNORE INTO memory_messages
                                (provider, project, conversation_id, role, content,
                                 timestamp, metadata_json)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    r.provider,
                                    r.project,
                                    r.conversation_id,
                                    r.role,
                                    r.content,
                                    r.timestamp,
                                    r.metadata_json,
                                ),
                            )
                            if not (cur.rowcount and cur.lastrowid):
                                continue
                            new_id = int(cur.lastrowid)

                        inserted += 1
                        if terms_ready:
                            index_message_terms(
                                conn, new_id, r.content, term_mapper=term_mapper
                            )
                    conn.commit()
                    return inserted
                except Exception:
                    conn.rollback()
                    raise

        return with_busy_retry(_do)

    def list_messages(self, project: str) -> list[dict]:
        self.encryption_context()  # cache before nested connection use
        with self.connection() as conn:
            cols = "id, provider, project, conversation_id, role, content, timestamp, metadata_json"
            if self._message_has_source_key_column(conn):
                cols += ", source_key"
            cur = conn.execute(
                f"""
                SELECT {cols}
                FROM memory_messages
                WHERE project = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (project,),
            )
            return [self._unseal_message_dict(r) for r in cur.fetchall()]

    def list_messages_by_provider(self, project: str, provider: str) -> list[dict]:
        self.encryption_context()
        with self.connection() as conn:
            cols = "id, provider, project, conversation_id, role, content, timestamp, metadata_json"
            if self._message_has_source_key_column(conn):
                cols += ", source_key"
            cur = conn.execute(
                f"""
                SELECT {cols}
                FROM memory_messages
                WHERE project = ? AND provider = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (project, provider),
            )
            return [self._unseal_message_dict(r) for r in cur.fetchall()]

    def list_messages_filtered(
        self,
        project: str,
        provider: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict]:
        """List messages, optionally filtered by provider and/or conversation id substring."""
        self.encryption_context()
        clauses = ["project = ?"]
        params: list[str] = [project]
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if conversation_id:
            clauses.append("conversation_id LIKE ?")
            params.append(f"%{conversation_id}%")
        where = " AND ".join(clauses)
        with self.connection() as conn:
            cols = "id, provider, project, conversation_id, role, content, timestamp, metadata_json"
            if self._message_has_source_key_column(conn):
                cols += ", source_key"
            cur = conn.execute(
                f"""
                SELECT {cols}
                FROM memory_messages
                WHERE {where}
                ORDER BY timestamp ASC, id ASC
                """,
                params,
            )
            return [self._unseal_message_dict(r) for r in cur.fetchall()]

    def list_conversations(
        self,
        project: str,
        provider: str | None = None,
        limit: int = 30,
    ) -> list[dict]:
        """Summarize conversations for a project (message count, last timestamp, preview)."""
        self.encryption_context()
        clauses = ["project = ?"]
        params: list[str | int] = [project]
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        where = " AND ".join(clauses)
        params.append(limit)
        with self.connection() as conn:
            # One round-trip: aggregate conversations and attach the latest user
            # preview via a windowed subquery (avoids N+1 preview lookups).
            cur = conn.execute(
                f"""
                WITH ranked AS (
                    SELECT
                        conversation_id,
                        provider,
                        COUNT(*) AS message_count,
                        MAX(timestamp) AS last_timestamp
                    FROM memory_messages
                    WHERE {where}
                    GROUP BY provider, conversation_id
                    ORDER BY last_timestamp DESC
                    LIMIT ?
                ),
                previews AS (
                    SELECT
                        provider,
                        conversation_id,
                        id AS preview_id,
                        content AS preview_content,
                        ROW_NUMBER() OVER (
                            PARTITION BY provider, conversation_id
                            ORDER BY timestamp DESC, id DESC
                        ) AS rn
                    FROM memory_messages
                    WHERE project = ?
                      AND role = 'user'
                )
                SELECT
                    ranked.conversation_id,
                    ranked.provider,
                    ranked.message_count,
                    ranked.last_timestamp,
                    previews.preview_id,
                    previews.preview_content
                FROM ranked
                LEFT JOIN previews
                  ON previews.provider = ranked.provider
                 AND previews.conversation_id = ranked.conversation_id
                 AND previews.rn = 1
                ORDER BY ranked.last_timestamp DESC
                """,
                [*params, project],
            )
            rows: list[dict] = []
            for raw in cur.fetchall():
                row = dict(raw)
                preview_id = row.pop("preview_id", None)
                preview_content = row.pop("preview_content", None)
                if preview_id is None or preview_content is None:
                    row["last_user_preview"] = ""
                else:
                    unsealed = self._unseal_message_dict(
                        {"id": preview_id, "content": preview_content}
                    )
                    row["last_user_preview"] = str(unsealed.get("content") or "")
                rows.append(row)
            return rows

    def list_agent_state(self, project: str) -> list[dict]:
        ctx = self.encryption_context()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT id, project, agent, state_key, value, updated_at
                FROM agent_state
                WHERE project = ?
                ORDER BY agent ASC, state_key ASC
                """,
                (project,),
            )
            out: list[dict] = []
            for row in cur.fetchall():
                value = str(row["value"])
                if ctx is not None:
                    from .crypto import decrypt_field, is_envelope

                    if is_envelope(value):
                        value = decrypt_field(
                            ctx,
                            value,
                            table="agent_state",
                            column="value",
                            row_identity=str(row["id"]),
                        )
                out.append(
                    {
                        "project": row["project"],
                        "agent": row["agent"],
                        "state_key": row["state_key"],
                        "value": value,
                        "updated_at": row["updated_at"],
                    }
                )
            return out

    def upsert_summary(self, rec: CompressedRecord) -> str:
        """Insert or update summary for (project, conversation_id).

        Returns ``\"inserted\"`` or ``\"updated\"``.
        When multiple historical duplicates exist, every matching row is updated
        deterministically (compatibility with the Batch 2 contract).
        """

        def _do() -> str:
            with self.connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    existing = conn.execute(
                        """
                        SELECT id FROM memory_summaries
                        WHERE project = ? AND conversation_id = ?
                        ORDER BY id ASC
                        """,
                        (rec.project, rec.conversation_id),
                    ).fetchall()
                    if not existing:
                        sid = self._next_row_id(conn, "memory_summaries")
                        store_summary = self._encrypt_field_value(
                            rec.summary,
                            table="memory_summaries",
                            column="summary",
                            row_identity=sid,
                        )
                        conn.execute(
                            """
                            INSERT INTO memory_summaries
                            (id, project, conversation_id, summary, method, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                sid,
                                rec.project,
                                rec.conversation_id,
                                store_summary,
                                rec.method,
                                rec.created_at,
                            ),
                        )
                        conn.commit()
                        return "inserted"

                    # Update every pre-existing duplicate row (encrypt before write).
                    for row in existing:
                        sid = int(row["id"])
                        store_summary = self._encrypt_field_value(
                            rec.summary,
                            table="memory_summaries",
                            column="summary",
                            row_identity=sid,
                        )
                        conn.execute(
                            """
                            UPDATE memory_summaries
                            SET summary = ?, method = ?, created_at = ?
                            WHERE id = ?
                            """,
                            (store_summary, rec.method, rec.created_at, sid),
                        )
                    conn.commit()
                    return "updated"
                except Exception:
                    conn.rollback()
                    raise

        return with_busy_retry(_do)

    def set_agent_state(self, rec: AgentState) -> None:
        def _do() -> None:
            with self.connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    row = conn.execute(
                        """
                        SELECT id FROM agent_state
                        WHERE project = ? AND agent = ? AND state_key = ?
                        """,
                        (rec.project, rec.agent, rec.key),
                    ).fetchone()
                    if row is None:
                        sid = self._next_row_id(conn, "agent_state")
                        store_value = self._encrypt_field_value(
                            rec.value,
                            table="agent_state",
                            column="value",
                            row_identity=sid,
                        )
                        conn.execute(
                            """
                            INSERT INTO agent_state
                            (id, project, agent, state_key, value, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                sid,
                                rec.project,
                                rec.agent,
                                rec.key,
                                store_value,
                                rec.updated_at,
                            ),
                        )
                    else:
                        sid = int(row["id"])
                        store_value = self._encrypt_field_value(
                            rec.value,
                            table="agent_state",
                            column="value",
                            row_identity=sid,
                        )
                        conn.execute(
                            """
                            UPDATE agent_state
                            SET value = ?, updated_at = ?
                            WHERE id = ?
                            """,
                            (store_value, rec.updated_at, sid),
                        )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

        with_busy_retry(_do)

    def get_agent_state(self, project: str, agent: str, key: str) -> str | None:
        ctx = self.encryption_context()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT id, value FROM agent_state
                WHERE project = ? AND agent = ? AND state_key = ?
                """,
                (project, agent, key),
            )
            row = cur.fetchone()
            if row is None:
                return None
            value = str(row["value"])
            if ctx is not None:
                from .crypto import decrypt_field, is_envelope

                if is_envelope(value):
                    value = decrypt_field(
                        ctx,
                        value,
                        table="agent_state",
                        column="value",
                        row_identity=str(row["id"]),
                    )
            return value

    def save_embedding(self, message_id: int, embedding_json: str) -> None:
        def _do() -> None:
            with self.connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    row = conn.execute(
                        "SELECT id FROM memory_embeddings WHERE message_id = ?",
                        (message_id,),
                    ).fetchone()
                    if row is None:
                        eid = self._next_row_id(conn, "memory_embeddings")
                        store = self._encrypt_field_value(
                            embedding_json,
                            table="memory_embeddings",
                            column="embedding_json",
                            row_identity=eid,
                        )
                        conn.execute(
                            """
                            INSERT INTO memory_embeddings (id, message_id, embedding_json)
                            VALUES (?, ?, ?)
                            """,
                            (eid, message_id, store),
                        )
                    else:
                        eid = int(row["id"])
                        store = self._encrypt_field_value(
                            embedding_json,
                            table="memory_embeddings",
                            column="embedding_json",
                            row_identity=eid,
                        )
                        conn.execute(
                            """
                            UPDATE memory_embeddings SET embedding_json = ? WHERE id = ?
                            """,
                            (store, eid),
                        )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

        with_busy_retry(_do)

    def list_summaries(self, project: str) -> list[dict]:
        self.encryption_context()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT id, conversation_id, summary, method, created_at
                FROM memory_summaries
                WHERE project = ?
                ORDER BY created_at DESC
                """,
                (project,),
            )
            return [self._unseal_summary_dict(r) for r in cur.fetchall()]

    def project_stats(self, project: str) -> dict[str, int]:
        with self.connection() as conn:
            msg_count = conn.execute(
                "SELECT COUNT(*) AS c FROM memory_messages WHERE project = ?",
                (project,),
            ).fetchone()["c"]
            conv_count = conn.execute(
                "SELECT COUNT(DISTINCT conversation_id) AS c FROM memory_messages WHERE project = ?",
                (project,),
            ).fetchone()["c"]
            sum_count = conn.execute(
                "SELECT COUNT(*) AS c FROM memory_summaries WHERE project = ?",
                (project,),
            ).fetchone()["c"]
            emb_count = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM memory_embeddings e
                JOIN memory_messages m ON e.message_id = m.id
                WHERE m.project = ?
                """,
                (project,),
            ).fetchone()["c"]
        return {
            "messages": int(msg_count),
            "conversations": int(conv_count),
            "summaries": int(sum_count),
            "embeddings": int(emb_count),
        }

    def count_embeddings(
        self,
        project: str,
        *,
        provider: str | None = None,
        conversation_id: str | None = None,
        role: str | None = None,
    ) -> int:
        """Count eligible embeddings in scope without loading vectors."""
        with self.connection() as conn:
            filters = ["m.project = ?"]
            params: list[object] = [project]
            if provider is not None:
                filters.append("m.provider = ?")
                params.append(provider)
            if conversation_id is not None:
                filters.append("m.conversation_id = ?")
                params.append(conversation_id)
            if role is not None:
                filters.append("m.role = ?")
                params.append(role)
            where = " AND ".join(filters)
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS c
                FROM memory_embeddings e
                JOIN memory_messages m ON e.message_id = m.id
                WHERE {where}
                """,
                params,
            ).fetchone()
            return int(row["c"])

    def list_embeddings(self, project: str) -> list[dict]:
        self.encryption_context()
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT m.id AS message_id, m.content, m.provider, m.conversation_id,
                       e.id AS embedding_id, e.embedding_json
                FROM memory_embeddings e
                JOIN memory_messages m ON e.message_id = m.id
                WHERE m.project = ?
                """,
                (project,),
            )
            return [self._unseal_embedding_dict(r) for r in cur.fetchall()]

    def list_embeddings_by_ids(
        self,
        message_ids: list[int],
        *,
        project: str,
        provider: str | None = None,
        conversation_id: str | None = None,
        role: str | None = None,
    ) -> list[dict]:
        """Fetch embeddings for candidate message ids within scope."""
        if not message_ids:
            return []
        self.encryption_context()
        with self.connection() as conn:
            placeholders = ",".join("?" for _ in message_ids)
            filters = [f"m.id IN ({placeholders})", "m.project = ?"]
            params: list[object] = list(message_ids)
            params.append(project)
            if provider is not None:
                filters.append("m.provider = ?")
                params.append(provider)
            if conversation_id is not None:
                filters.append("m.conversation_id = ?")
                params.append(conversation_id)
            if role is not None:
                filters.append("m.role = ?")
                params.append(role)
            where = " AND ".join(filters)
            cur = conn.execute(
                f"""
                SELECT m.id AS message_id, m.content, m.provider,
                       m.conversation_id, e.id AS embedding_id, e.embedding_json
                FROM memory_embeddings e
                JOIN memory_messages m ON e.message_id = m.id
                WHERE {where}
                """,
                params,
            )
            return [self._unseal_embedding_dict(r) for r in cur.fetchall()]

    def find_message_id_by_exact_content(
        self,
        *,
        project: str,
        provider: str,
        conversation_id: str,
        role: str,
        content: str,
    ) -> int | None:
        """Return an existing message id for exact-content dedupe."""
        ctx = self.encryption_context()
        with self.connection() as conn:
            if ctx is not None and self._has_content_fingerprint_column(conn):
                from .crypto import content_fingerprint

                fp = content_fingerprint(
                    ctx,
                    project=project,
                    provider=provider,
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                )
                cur = conn.execute(
                    """
                    SELECT id FROM memory_messages
                    WHERE project = ? AND provider = ? AND conversation_id = ?
                      AND role = ? AND content_fingerprint = ?
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    (project, provider, conversation_id, role, fp),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT id FROM memory_messages
                    WHERE project = ? AND provider = ? AND conversation_id = ?
                      AND role = ? AND content = ?
                    ORDER BY id ASC
                    LIMIT 1
                    """,
                    (project, provider, conversation_id, role, content),
                )
            row = cur.fetchone()
            return None if row is None else int(row["id"])

    def list_messages_for_namespace(
        self,
        *,
        project: str,
        provider: str,
        conversation_id: str,
        role: str | None = None,
    ) -> list[dict]:
        self.encryption_context()
        with self.connection() as conn:
            cols = "id, provider, project, conversation_id, role, content, timestamp, metadata_json"
            if self._message_has_source_key_column(conn):
                cols += ", source_key"
            if role is None:
                cur = conn.execute(
                    f"""
                    SELECT {cols}
                    FROM memory_messages
                    WHERE project = ? AND provider = ? AND conversation_id = ?
                    ORDER BY timestamp ASC, id ASC
                    """,
                    (project, provider, conversation_id),
                )
            else:
                cur = conn.execute(
                    f"""
                    SELECT {cols}
                    FROM memory_messages
                    WHERE project = ? AND provider = ? AND conversation_id = ?
                      AND role = ?
                    ORDER BY timestamp ASC, id ASC
                    """,
                    (project, provider, conversation_id, role),
                )
            return [self._unseal_message_dict(r) for r in cur.fetchall()]

    def list_embeddings_for_namespace(
        self,
        *,
        project: str,
        provider: str,
        conversation_id: str,
        role: str | None = None,
    ) -> list[dict]:
        self.encryption_context()
        with self.connection() as conn:
            if role is None:
                cur = conn.execute(
                    """
                    SELECT m.id AS message_id, m.content, m.provider,
                           m.conversation_id, e.id AS embedding_id, e.embedding_json
                    FROM memory_embeddings e
                    JOIN memory_messages m ON e.message_id = m.id
                    WHERE m.project = ? AND m.provider = ?
                      AND m.conversation_id = ?
                    """,
                    (project, provider, conversation_id),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT m.id AS message_id, m.content, m.provider,
                           m.conversation_id, e.id AS embedding_id, e.embedding_json
                    FROM memory_embeddings e
                    JOIN memory_messages m ON e.message_id = m.id
                    WHERE m.project = ? AND m.provider = ?
                      AND m.conversation_id = ? AND m.role = ?
                    """,
                    (project, provider, conversation_id, role),
                )
            return [self._unseal_embedding_dict(r) for r in cur.fetchall()]

    def list_all_projects(self) -> list[str]:
        """Return distinct project namespaces in this database."""
        with self.connection() as conn:
            cur = conn.execute(
                "SELECT DISTINCT project FROM memory_messages ORDER BY project ASC"
            )
            return [str(row["project"]) for row in cur.fetchall()]

    def list_embeddings_cross_project(
        self,
        *,
        provider: str | None = None,
        role: str | None = None,
    ) -> list[dict]:
        """Fetch embeddings across all projects for mesh search."""
        self.encryption_context()
        clauses: list[str] = []
        params: list[object] = []
        if provider is not None:
            clauses.append("m.provider = ?")
            params.append(provider)
        if role is not None:
            clauses.append("m.role = ?")
            params.append(role)
        where = (" AND ".join(clauses)) if clauses else "1=1"
        with self.connection() as conn:
            cur = conn.execute(
                f"""
                SELECT m.id AS message_id, m.content, m.provider, m.project,
                       m.conversation_id, e.id AS embedding_id, e.embedding_json
                FROM memory_embeddings e
                JOIN memory_messages m ON e.message_id = m.id
                WHERE {where}
                """,
                params,
            )
            return [self._unseal_embedding_dict(r) for r in cur.fetchall()]

    def list_embeddings_by_ids_cross_project(
        self,
        message_ids: list[int],
        *,
        provider: str | None = None,
        role: str | None = None,
    ) -> list[dict]:
        """Fetch embeddings for candidate message ids across all projects."""
        if not message_ids:
            return []
        self.encryption_context()
        placeholders = ",".join("?" for _ in message_ids)
        clauses = [f"m.id IN ({placeholders})"]
        params: list[object] = list(message_ids)
        if provider is not None:
            clauses.append("m.provider = ?")
            params.append(provider)
        if role is not None:
            clauses.append("m.role = ?")
            params.append(role)
        where = " AND ".join(clauses)
        with self.connection() as conn:
            cur = conn.execute(
                f"""
                SELECT m.id AS message_id, m.content, m.provider, m.project,
                       m.conversation_id, e.id AS embedding_id, e.embedding_json
                FROM memory_embeddings e
                JOIN memory_messages m ON e.message_id = m.id
                WHERE {where}
                """,
                params,
            )
            return [self._unseal_embedding_dict(r) for r in cur.fetchall()]

    def count_embeddings_cross_project(
        self,
        *,
        provider: str | None = None,
        role: str | None = None,
    ) -> int:
        """Count eligible embeddings across all projects."""
        with self.connection() as conn:
            clauses: list[str] = []
            params: list[object] = []
            if provider is not None:
                clauses.append("m.provider = ?")
                params.append(provider)
            if role is not None:
                clauses.append("m.role = ?")
                params.append(role)
            where = (" AND ".join(clauses)) if clauses else "1=1"
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS c
                FROM memory_embeddings e
                JOIN memory_messages m ON e.message_id = m.id
                WHERE {where}
                """,
                params,
            ).fetchone()
            return int(row["c"])

    def select_candidate_message_ids_cross_project(
        self,
        *,
        terms: list[str],
        limit: int,
        provider: str | None = None,
        role: str | None = None,
    ) -> list[int]:
        """Rank message ids by term overlap across all projects."""
        from .search_index import select_candidate_message_ids

        if not terms or limit <= 0:
            return []
        self.encryption_context()
        placeholders = ",".join("?" for _ in terms)
        clauses = [f"t.term IN ({placeholders})"]
        params: list[object] = list(terms)
        if provider is not None:
            clauses.append("m.provider = ?")
            params.append(provider)
        if role is not None:
            clauses.append("m.role = ?")
            params.append(role)
        clauses.append(
            "EXISTS (SELECT 1 FROM memory_embeddings e WHERE e.message_id = m.id)"
        )
        where = " AND ".join(clauses)
        params.append(int(limit))
        with self.connection() as conn:
            cur = conn.execute(
                f"""
                SELECT t.message_id AS message_id, COUNT(DISTINCT t.term) AS overlap
                FROM memory_message_terms t
                JOIN memory_messages m ON m.id = t.message_id
                WHERE {where}
                GROUP BY t.message_id
                ORDER BY overlap DESC, t.message_id ASC
                LIMIT ?
                """,
                params,
            )
            return [int(r[0]) for r in cur.fetchall()]

    def insert_message_returning_id(self, rec: MemoryRecord) -> int | None:
        """Insert one message; return new id, or None if UNIQUE dedupe skipped it."""

        def _do() -> int | None:
            ctx = self.encryption_context()
            term_mapper = self._term_mapper()
            encrypted = ctx is not None
            with self.connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    has_sk = self._message_has_source_key_column(conn)
                    if encrypted:
                        new_id = self._next_row_id(conn, "memory_messages")
                        store_content, store_meta, store_fp = self._encrypt_message_fields(
                            new_id,
                            project=rec.project,
                            provider=rec.provider,
                            conversation_id=rec.conversation_id,
                            role=rec.role,
                            content=rec.content,
                            metadata_json=rec.metadata_json,
                        )
                        cols = (
                            "id, provider, project, conversation_id, role, content, "
                            "timestamp, metadata_json"
                        )
                        vals: list[object] = [
                            new_id,
                            rec.provider,
                            rec.project,
                            rec.conversation_id,
                            rec.role,
                            store_content,
                            rec.timestamp,
                            store_meta,
                        ]
                        if has_sk:
                            cols += ", source_key"
                            vals.append(rec.source_key)
                        if self._has_content_fingerprint_column(conn) and store_fp is not None:
                            cols += ", content_fingerprint"
                            vals.append(store_fp)
                        placeholders = ", ".join("?" * len(vals))
                        cur = conn.execute(
                            f"INSERT OR IGNORE INTO memory_messages ({cols}) VALUES ({placeholders})",
                            tuple(vals),
                        )
                        if cur.rowcount == 0:
                            conn.commit()
                            return None
                    elif has_sk:
                        cur = conn.execute(
                            """
                            INSERT OR IGNORE INTO memory_messages
                            (provider, project, conversation_id, role, content,
                             timestamp, metadata_json, source_key)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                rec.provider,
                                rec.project,
                                rec.conversation_id,
                                rec.role,
                                rec.content,
                                rec.timestamp,
                                rec.metadata_json,
                                rec.source_key,
                            ),
                        )
                        if cur.rowcount == 0:
                            conn.commit()
                            return None
                        new_id = int(cur.lastrowid)
                    else:
                        cur = conn.execute(
                            """
                            INSERT OR IGNORE INTO memory_messages
                            (provider, project, conversation_id, role, content,
                             timestamp, metadata_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                rec.provider,
                                rec.project,
                                rec.conversation_id,
                                rec.role,
                                rec.content,
                                rec.timestamp,
                                rec.metadata_json,
                            ),
                        )
                        if cur.rowcount == 0:
                            conn.commit()
                            return None
                        new_id = int(cur.lastrowid)
                    if self._terms_table_ready(conn):
                        index_message_terms(
                            conn, new_id, rec.content, term_mapper=term_mapper
                        )
                    conn.commit()
                    return new_id
                except Exception:
                    conn.rollback()
                    raise

        return with_busy_retry(_do)

    def store_facade_memory_with_embedding(
        self,
        *,
        rec: MemoryRecord,
        embedding_json: str,
    ) -> tuple[str, int]:
        """Atomically dedupe + insert facade memory and its embedding.

        Uses ``BEGIN IMMEDIATE`` so concurrent ``Memory`` instances cannot both
        insert the same exact-content facade row. Returns
        ``(\"inserted\", message_id)`` or ``(\"duplicate\", existing_id)``.
        Duplicates leave the existing embedding unchanged.
        """

        def _do() -> tuple[str, int]:
            ctx = self.encryption_context()
            term_mapper = self._term_mapper()
            with self.connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    if ctx is not None and self._has_content_fingerprint_column(conn):
                        from .crypto import content_fingerprint

                        fp = content_fingerprint(
                            ctx,
                            project=rec.project,
                            provider=rec.provider,
                            conversation_id=rec.conversation_id,
                            role=rec.role,
                            content=rec.content,
                        )
                        cur = conn.execute(
                            """
                            SELECT id FROM memory_messages
                            WHERE project = ? AND provider = ? AND conversation_id = ?
                              AND role = ? AND content_fingerprint = ?
                            ORDER BY id ASC
                            LIMIT 1
                            """,
                            (
                                rec.project,
                                rec.provider,
                                rec.conversation_id,
                                rec.role,
                                fp,
                            ),
                        )
                    else:
                        cur = conn.execute(
                            """
                            SELECT id FROM memory_messages
                            WHERE project = ? AND provider = ? AND conversation_id = ?
                              AND role = ? AND content = ?
                            ORDER BY id ASC
                            LIMIT 1
                            """,
                            (
                                rec.project,
                                rec.provider,
                                rec.conversation_id,
                                rec.role,
                                rec.content,
                            ),
                        )
                    row = cur.fetchone()
                    if row is not None:
                        existing_id = int(row["id"])
                        conn.commit()
                        return "duplicate", existing_id

                    has_sk = self._message_has_source_key_column(conn)
                    if ctx is not None:
                        new_id = self._next_row_id(conn, "memory_messages")
                        store_content, store_meta, store_fp = self._encrypt_message_fields(
                            new_id,
                            project=rec.project,
                            provider=rec.provider,
                            conversation_id=rec.conversation_id,
                            role=rec.role,
                            content=rec.content,
                            metadata_json=rec.metadata_json,
                        )
                        cols = (
                            "id, provider, project, conversation_id, role, content, "
                            "timestamp, metadata_json"
                        )
                        vals: list[object] = [
                            new_id,
                            rec.provider,
                            rec.project,
                            rec.conversation_id,
                            rec.role,
                            store_content,
                            rec.timestamp,
                            store_meta,
                        ]
                        if has_sk:
                            cols += ", source_key"
                            vals.append(rec.source_key)
                        if self._has_content_fingerprint_column(conn) and store_fp is not None:
                            cols += ", content_fingerprint"
                            vals.append(store_fp)
                        placeholders = ", ".join("?" * len(vals))
                        conn.execute(
                            f"INSERT INTO memory_messages ({cols}) VALUES ({placeholders})",
                            tuple(vals),
                        )
                    elif has_sk:
                        cur = conn.execute(
                            """
                            INSERT INTO memory_messages
                            (provider, project, conversation_id, role, content,
                             timestamp, metadata_json, source_key)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                rec.provider,
                                rec.project,
                                rec.conversation_id,
                                rec.role,
                                rec.content,
                                rec.timestamp,
                                rec.metadata_json,
                                rec.source_key,
                            ),
                        )
                        new_id = int(cur.lastrowid)
                    else:
                        cur = conn.execute(
                            """
                            INSERT INTO memory_messages
                            (provider, project, conversation_id, role, content,
                             timestamp, metadata_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                rec.provider,
                                rec.project,
                                rec.conversation_id,
                                rec.role,
                                rec.content,
                                rec.timestamp,
                                rec.metadata_json,
                            ),
                        )
                        new_id = int(cur.lastrowid)
                    # Embedding: encrypt before insert using preallocated id.
                    emb_row = conn.execute(
                        "SELECT id FROM memory_embeddings WHERE message_id = ?",
                        (new_id,),
                    ).fetchone()
                    if emb_row is None:
                        eid = self._next_row_id(conn, "memory_embeddings")
                        store_emb = self._encrypt_field_value(
                            embedding_json,
                            table="memory_embeddings",
                            column="embedding_json",
                            row_identity=eid,
                        )
                        conn.execute(
                            """
                            INSERT INTO memory_embeddings (id, message_id, embedding_json)
                            VALUES (?, ?, ?)
                            """,
                            (eid, new_id, store_emb),
                        )
                    else:
                        eid = int(emb_row["id"])
                        store_emb = self._encrypt_field_value(
                            embedding_json,
                            table="memory_embeddings",
                            column="embedding_json",
                            row_identity=eid,
                        )
                        conn.execute(
                            """
                            UPDATE memory_embeddings SET embedding_json = ? WHERE id = ?
                            """,
                            (store_emb, eid),
                        )
                    if self._terms_table_ready(conn):
                        index_message_terms(
                            conn, new_id, rec.content, term_mapper=term_mapper
                        )
                    conn.commit()
                    return "inserted", new_id
                except Exception:
                    conn.rollback()
                    raise

        return with_busy_retry(_do)
