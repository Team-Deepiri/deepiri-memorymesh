"""Ordered, transactional schema migrations for the canonical platform DB (T16).

Version history is authoritative in ``schema_migrations``. ``PRAGMA user_version``
is kept as a synchronized mirror only and must never silently disagree.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

# Keep in sync with storage.BUSY_TIMEOUT_MS when both are configured.
DEFAULT_BUSY_TIMEOUT_MS = 5000

MigrationFn = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    apply: MigrationFn


@dataclass(slots=True)
class MigrationStatus:
    """Read-only migration status (never mutates the database)."""

    db_path: Path
    current_version: int
    latest_version: int
    pending: list[tuple[int, str]] = field(default_factory=list)
    journal_mode: str | None = None
    nonempty: bool = False
    adopted_unversioned_baseline: bool = False


@dataclass(slots=True)
class MigrationReport:
    """Result of a migrate/apply call."""

    db_path: Path
    from_version: int
    to_version: int
    applied: list[tuple[int, str]] = field(default_factory=list)
    pending: list[tuple[int, str]] = field(default_factory=list)
    backup_path: Path | None = None
    dry_run: bool = False
    no_change: bool = False
    adopted_unversioned_baseline: bool = False
    message: str = ""


class MigrationError(RuntimeError):
    """Schema migration or compatibility failure."""

    def __init__(self, message: str, *, version: int | None = None, name: str | None = None):
        self.version = version
        self.name = name
        super().__init__(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return {str(row[0]) for row in cur.fetchall()}


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _index_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
    )
    return {str(row[0]) for row in cur.fetchall()}


def _is_exact_batch4_baseline(conn: sqlite3.Connection) -> bool:
    """True only when the DB matches the Batch 4 unversioned platform shape exactly."""
    tables = _table_names(conn)
    expected_tables = {
        "memory_messages",
        "memory_summaries",
        "memory_embeddings",
        "agent_state",
    }
    # Allow sqlite_sequence; reject anything else (including schema_migrations,
    # source_key era tables, or unrelated extras).
    extras = tables - expected_tables - {"sqlite_sequence"}
    if extras or not expected_tables.issubset(tables):
        return False

    msg_cols = {
        "id",
        "provider",
        "project",
        "conversation_id",
        "role",
        "content",
        "timestamp",
        "metadata_json",
    }
    if _column_names(conn, "memory_messages") != msg_cols:
        return False
    if _column_names(conn, "memory_summaries") != {
        "id",
        "project",
        "conversation_id",
        "summary",
        "method",
        "created_at",
    }:
        return False
    if _column_names(conn, "memory_embeddings") != {
        "id",
        "message_id",
        "embedding_json",
    }:
        return False
    if _column_names(conn, "agent_state") != {
        "id",
        "project",
        "agent",
        "state_key",
        "value",
        "updated_at",
    }:
        return False

    indexes = _index_names(conn)
    # Required unique indexes from Batch 4 SCHEMA. Agent unique may be
    # sqlite-auto-named from UNIQUE(...); require the two named indexes.
    if "ux_messages_dedupe" not in indexes:
        return False
    if "ux_embeddings_message" not in indexes:
        return False
    return True


def _baseline_ddl(conn: sqlite3.Connection) -> None:
    """Create the Batch 4 / version-1 canonical platform schema.

    Uses discrete ``execute`` calls (not ``executescript``) so an outer
    ``BEGIN IMMEDIATE`` transaction is not implicitly committed.
    """
    statements = [
        """
        CREATE TABLE IF NOT EXISTS memory_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            project TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_messages_dedupe
        ON memory_messages (project, provider, conversation_id, role, content, timestamp)
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            method TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            embedding_json TEXT NOT NULL,
            FOREIGN KEY(message_id) REFERENCES memory_messages(id)
        )
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_embeddings_message
        ON memory_embeddings (message_id)
        """,
        """
        CREATE TABLE IF NOT EXISTS agent_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project TEXT NOT NULL,
            agent TEXT NOT NULL,
            state_key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(project, agent, state_key)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """,
    ]
    for stmt in statements:
        conn.execute(stmt)


def _migrate_v1_baseline(conn: sqlite3.Connection) -> None:
    _baseline_ddl(conn)


def _migrate_v2_source_key(conn: sqlite3.Connection) -> None:
    cols = _column_names(conn, "memory_messages")
    if "source_key" not in cols:
        conn.execute("ALTER TABLE memory_messages ADD COLUMN source_key TEXT NULL")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_messages_source_key
        ON memory_messages (project, provider, conversation_id, source_key)
        WHERE source_key IS NOT NULL
        """
    )


def _migrate_v3_terms_index(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_message_terms (
            message_id INTEGER NOT NULL,
            term TEXT NOT NULL,
            PRIMARY KEY (message_id, term),
            FOREIGN KEY (message_id) REFERENCES memory_messages(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_message_terms_term
        ON memory_message_terms (term)
        """
    )
    from .search_index import backfill_message_terms

    backfill_message_terms(conn)


def _migrate_v4_encryption_security_metadata(conn: sqlite3.Connection) -> None:
    """Add the ``encryption_meta`` singleton row and message fingerprint column (T33).

    Plaintext-mode behavior is unchanged: ``enabled`` defaults to 0 and
    ``terms_mode`` defaults to ``'plaintext'``. ``db_identity`` is generated
    once here (random) and used later to bind AEAD ciphertext to this
    specific database instance.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS encryption_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            enabled INTEGER NOT NULL DEFAULT 0,
            key_id TEXT,
            algorithm TEXT,
            envelope_version INTEGER,
            db_identity TEXT NOT NULL,
            terms_mode TEXT NOT NULL DEFAULT 'plaintext',
            updated_at TEXT NOT NULL
        )
        """
    )
    existing = conn.execute("SELECT COUNT(*) FROM encryption_meta WHERE id = 1").fetchone()[0]
    if not existing:
        conn.execute(
            """
            INSERT INTO encryption_meta
                (id, enabled, key_id, algorithm, envelope_version, db_identity, terms_mode, updated_at)
            VALUES (1, 0, NULL, NULL, NULL, ?, 'plaintext', ?)
            """,
            (uuid.uuid4().hex, _utc_now()),
        )

    cols = _column_names(conn, "memory_messages")
    if "content_fingerprint" not in cols:
        conn.execute("ALTER TABLE memory_messages ADD COLUMN content_fingerprint TEXT NULL")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_messages_content_fingerprint
        ON memory_messages (project, provider, conversation_id, role, content_fingerprint)
        WHERE content_fingerprint IS NOT NULL
        """
    )


def _migrate_v5_project_http_tokens(conn: sqlite3.Connection) -> None:
    """Add ``project_http_tokens`` for scoped HTTP pull/push API auth."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_http_tokens (
            token_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            token_hash TEXT NOT NULL,
            label TEXT,
            scopes TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            revoked_at TEXT,
            last_used_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_project_http_tokens_project
        ON project_http_tokens (project)
        """
    )


def _migrate_v6_api_pull_state(conn: sqlite3.Connection) -> None:
    """Add ``api_pull_state`` for conditional-request bookkeeping on API pulls."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_pull_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url_hash TEXT NOT NULL,
            project TEXT NOT NULL,
            format TEXT NOT NULL,
            provider TEXT,
            etag TEXT,
            last_modified TEXT,
            last_success_at TEXT,
            UNIQUE(url_hash, project, format)
        )
        """
    )


MIGRATIONS: list[Migration] = [
    Migration(1, "baseline_platform_schema", _migrate_v1_baseline),
    Migration(2, "message_source_key", _migrate_v2_source_key),
    Migration(3, "message_terms_index", _migrate_v3_terms_index),
    Migration(4, "encryption_security_metadata", _migrate_v4_encryption_security_metadata),
    Migration(5, "project_http_tokens", _migrate_v5_project_http_tokens),
    Migration(6, "api_pull_state", _migrate_v6_api_pull_state),
]

CURRENT_SCHEMA_VERSION = MIGRATIONS[-1].version

_MIGRATIONS_BY_VERSION = {m.version: m for m in MIGRATIONS}


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def _read_migration_rows(conn: sqlite3.Connection) -> dict[int, str]:
    tables = _table_names(conn)
    if "schema_migrations" not in tables:
        return {}
    rows = conn.execute(
        "SELECT version, name FROM schema_migrations ORDER BY version ASC"
    ).fetchall()
    out: dict[int, str] = {}
    for version, name in rows:
        out[int(version)] = str(name)
    return out


def _validate_history(rows: dict[int, str]) -> int:
    """Validate contiguous history from 1..N with expected names; return current version."""
    if not rows:
        return 0
    versions = sorted(rows)
    if versions[0] != 1:
        raise MigrationError(
            f"Migration history must start at version 1; found {versions[0]}",
            version=versions[0],
        )
    expected = list(range(1, versions[-1] + 1))
    if versions != expected:
        raise MigrationError(
            f"Migration history has gaps or duplicates: {versions}"
        )
    for ver in versions:
        mig = _MIGRATIONS_BY_VERSION.get(ver)
        if mig is None:
            raise MigrationError(
                f"Unsupported future or unknown schema version {ver} in history",
                version=ver,
            )
        if rows[ver] != mig.name:
            raise MigrationError(
                f"Migration history name mismatch at version {ver}: "
                f"recorded={rows[ver]!r} expected={mig.name!r}",
                version=ver,
                name=rows[ver],
            )
    current = versions[-1]
    if current > CURRENT_SCHEMA_VERSION:
        raise MigrationError(
            f"Database schema version {current} is newer than supported "
            f"{CURRENT_SCHEMA_VERSION}",
            version=current,
        )
    return current


def _sync_user_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(f"PRAGMA user_version = {int(version)}")


def _read_user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _assert_user_version_mirror(conn: sqlite3.Connection, current: int) -> None:
    mirrored = _read_user_version(conn)
    if mirrored != current:
        raise MigrationError(
            f"PRAGMA user_version ({mirrored}) disagrees with "
            f"schema_migrations current version ({current})"
        )


def current_schema_version(conn: sqlite3.Connection) -> int:
    """Return the applied schema version (0 if unversioned/empty)."""
    rows = _read_migration_rows(conn)
    if rows:
        current = _validate_history(rows)
        _assert_user_version_mirror(conn, current)
        return current
    # Unversioned: may still be exact Batch 4 baseline (treated as needing adoption).
    return 0


def pending_migrations(conn: sqlite3.Connection) -> list[Migration]:
    current = current_schema_version(conn)
    return [m for m in MIGRATIONS if m.version > current]


def _db_row_count(conn: sqlite3.Connection) -> int:
    total = 0
    for table in (
        "memory_messages",
        "memory_summaries",
        "memory_embeddings",
        "agent_state",
    ):
        if table in _table_names(conn):
            total += int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    return total


def _configure_conn(conn: sqlite3.Connection, *, busy_timeout_ms: int) -> None:
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    # Align Python sqlite3 waiting with the PRAGMA.
    try:
        conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    except sqlite3.Error:
        pass


def _open_rw(db_path: Path, *, busy_timeout_ms: int) -> sqlite3.Connection:
    timeout_s = max(busy_timeout_ms / 1000.0, 0.001)
    conn = sqlite3.connect(db_path, timeout=timeout_s)
    _configure_conn(conn, busy_timeout_ms=busy_timeout_ms)
    return conn


def _checkpoint_wal(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("PRAGMA wal_checkpoint(FULL)")
    except sqlite3.Error:
        pass


def _create_backup(
    db_path: Path,
    conn: sqlite3.Connection,
    from_version: int,
    *,
    checkpoint: bool = False,
) -> Path:
    """Create a consistent SQLite backup; never overwrite an existing path.

    Uses the caller's locked connection so concurrent migrators cannot both
    snapshot. Do not run ``wal_checkpoint`` while holding ``BEGIN IMMEDIATE`` —
    that can deadlock. Checkpoint only when the caller is not in a write txn.
    """
    if checkpoint:
        _checkpoint_wal(conn)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = db_path.with_name(f"{db_path.name}.pre-migrate-v{from_version}.{stamp}.bak")
    if candidate.exists():
        raise MigrationError(f"Backup path already exists: {candidate}")
    dest = sqlite3.connect(candidate)
    try:
        conn.backup(dest)
        dest.commit()
    finally:
        dest.close()
    return candidate


def _enable_wal_if_file(conn: sqlite3.Connection, db_path: Path) -> str:
    """Enable WAL for file-backed DBs during init/migrate; return journal mode."""
    if str(db_path) == ":memory:" or db_path.name == "":
        mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        return mode
    # In-memory via shared cache URIs still report memory.
    try:
        result = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        return str(result[0]).lower()
    except sqlite3.Error:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()


def _classify_for_migrate(conn: sqlite3.Connection, db_path: Path) -> tuple[int, bool]:
    """Return (current_version, needs_unversioned_adoption).

    Raises MigrationError for incompatible schemas. Does not write.
    Uses only *conn* (must already hold a write lock for migrate) so concurrent
    migrators never observe a torn side-channel snapshot.
    """
    tables = _table_names(conn)
    rows = _read_migration_rows(conn)
    if rows:
        return _validate_history(rows), False

    if not tables:
        return 0, False

    has_platform = "memory_messages" in tables
    has_memories = "memories" in tables
    if has_platform and has_memories:
        raise MigrationError(
            f"Database at {db_path} has incompatible schema kind 'unknown'"
        )
    if has_memories and not has_platform:
        from .storage import _is_exact_legacy_memories_table

        extras = tables - {"memories", "sqlite_sequence"}
        if extras or not _is_exact_legacy_memories_table(conn):
            raise MigrationError(
                f"Database at {db_path} has incompatible schema kind 'unknown'"
            )
        raise MigrationError(
            f"Database at {db_path} uses the legacy simple-memory schema and "
            "cannot be migrated in place"
        )
    if has_platform:
        if _is_exact_batch4_baseline(conn):
            return 0, True
        raise MigrationError(
            f"Database at {db_path} has a partially recognized or altered "
            "platform schema without valid migration history"
        )
    raise MigrationError(
        f"Database at {db_path} has incompatible schema kind 'unknown'"
    )


def migration_status(
    db_path: Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> MigrationStatus:
    """Read-only status. Creates no backup and writes nothing."""
    path = Path(db_path)
    if not path.exists() or path.stat().st_size == 0:
        pending = [(m.version, m.name) for m in MIGRATIONS]
        return MigrationStatus(
            db_path=path,
            current_version=0,
            latest_version=CURRENT_SCHEMA_VERSION,
            pending=pending,
            journal_mode=None,
            nonempty=False,
        )

    # Read-only URI so status never creates journals.
    from .storage import sqlite_readonly_uri

    try:
        conn = sqlite3.connect(sqlite_readonly_uri(path), uri=True, timeout=busy_timeout_ms / 1000.0)
    except sqlite3.Error as exc:
        raise MigrationError(f"Cannot open database for status: {exc}") from exc
    try:
        conn.execute("PRAGMA query_only = ON")
        current, adopt = _classify_for_migrate(conn, path)
        pending = [(m.version, m.name) for m in MIGRATIONS if m.version > current]
        if adopt and current == 0:
            # Adoption records v1 then applies later versions.
            pending = [(m.version, m.name) for m in MIGRATIONS]
        journal = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        nonempty = _db_row_count(conn) > 0
        return MigrationStatus(
            db_path=path,
            current_version=current,
            latest_version=CURRENT_SCHEMA_VERSION,
            pending=pending,
            journal_mode=journal,
            nonempty=nonempty,
            adopted_unversioned_baseline=adopt,
        )
    finally:
        conn.close()


def migrate(
    db_path: Path,
    *,
    dry_run: bool = False,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    create_backup: bool = True,
) -> MigrationReport:
    """Apply pending migrations transactionally.

    Dry-run reports pending work without writing, backing up, or enabling WAL.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        status = migration_status(path, busy_timeout_ms=busy_timeout_ms)
        no_change = not status.pending
        return MigrationReport(
            db_path=path,
            from_version=status.current_version,
            to_version=status.current_version if no_change else CURRENT_SCHEMA_VERSION,
            applied=[],
            pending=list(status.pending),
            backup_path=None,
            dry_run=True,
            no_change=no_change,
            adopted_unversioned_baseline=status.adopted_unversioned_baseline,
            message="dry-run: no changes were made",
        )

    conn = _open_rw(path, busy_timeout_ms=busy_timeout_ms)
    backup_path: Path | None = None
    applied: list[tuple[int, str]] = []
    adopted = False
    from_version = 0
    current = 0
    try:
        # Acquire ownership before any schema work. Do not enable WAL first —
        # journal_mode changes contend with concurrent migrators and can time
        # out before BEGIN IMMEDIATE. WAL is enabled after the write txn.
        conn.execute("BEGIN IMMEDIATE")
        try:
            current, needs_adopt = _classify_for_migrate(conn, path)
            from_version = current
            nonempty = _db_row_count(conn) > 0
            pending = [m for m in MIGRATIONS if m.version > current]
            if needs_adopt:
                pending = list(MIGRATIONS)  # v1 adoption + later
            if not pending and not needs_adopt:
                conn.commit()
                _enable_wal_if_file(conn, path)
                return MigrationReport(
                    db_path=path,
                    from_version=from_version,
                    to_version=current,
                    applied=[],
                    pending=[],
                    backup_path=None,
                    dry_run=False,
                    no_change=True,
                    adopted_unversioned_baseline=False,
                    message="already at latest schema version",
                )

            # Backup while still holding BEGIN IMMEDIATE ownership. Use a
            # read-only side connection so sqlite3.backup does not deadlock with
            # the write transaction. Other migrators remain blocked on IMMEDIATE.
            # Callers that discover no pending work after locking never reach here.
            if create_backup and nonempty:
                from .storage import sqlite_readonly_uri

                backup_src = sqlite3.connect(
                    sqlite_readonly_uri(path),
                    uri=True,
                    timeout=busy_timeout_ms / 1000.0,
                )
                try:
                    backup_path = _create_backup(
                        path, backup_src, from_version, checkpoint=False
                    )
                finally:
                    backup_src.close()

            if needs_adopt:
                _ensure_migrations_table(conn)
                mig1 = _MIGRATIONS_BY_VERSION[1]
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (mig1.version, mig1.name, _utc_now()),
                )
                _sync_user_version(conn, 1)
                current = 1
                adopted = True
                applied.append((mig1.version, mig1.name))
                pending = [m for m in MIGRATIONS if m.version > 1]

            for mig in pending:
                try:
                    mig.apply(conn)
                except Exception as exc:
                    raise MigrationError(
                        f"Migration {mig.version} ({mig.name}) failed: {exc}",
                        version=mig.version,
                        name=mig.name,
                    ) from exc
                _ensure_migrations_table(conn)
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                    (mig.version, mig.name, _utc_now()),
                )
                _sync_user_version(conn, mig.version)
                applied.append((mig.version, mig.name))
                current = mig.version

            conn.commit()
        except Exception:
            conn.rollback()
            raise

        # WAL must be set outside a write transaction.
        _enable_wal_if_file(conn, path)
        _checkpoint_wal(conn)

        return MigrationReport(
            db_path=path,
            from_version=from_version,
            to_version=current,
            applied=applied,
            pending=[],
            backup_path=backup_path,
            dry_run=False,
            no_change=False,
            adopted_unversioned_baseline=adopted,
            message="ok",
        )
    finally:
        conn.close()


def ensure_migrated(
    db_path: Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> MigrationReport:
    """Idempotent migrate used by MemoryStore.init()."""
    return migrate(db_path, dry_run=False, busy_timeout_ms=busy_timeout_ms)
