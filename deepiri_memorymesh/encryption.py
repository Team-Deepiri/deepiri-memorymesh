"""Enable/rotate/status orchestration for field-level encryption (T33).

This module wires the primitives in :mod:`crypto` into the canonical
platform SQLite store. It owns:

- reading/writing the ``encryption_meta`` singleton row (created by
  migration v4, see :mod:`migrations`)
- ``enable_encryption``: first-time encryption of sensitive columns
- ``rotate_encryption``: re-encrypting under a new key
- ``encryption_status``: read-only diagnostics

When encryption has never been enabled, none of this module's write paths
run and all existing plaintext behavior is unchanged — ``encryption_meta``
simply reports ``enabled=False`` / ``terms_mode='plaintext'``.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .crypto import (
    ALGORITHM,
    CRYPTOGRAPHY_AVAILABLE,
    CryptoError,
    ENVELOPE_VERSION,
    EncryptionContext,
    WrongKeyError,
    content_fingerprint,
    decrypt_field,
    encrypt_field,
    keyed_term_token,
    require_cryptography,
)
from .migrations import DEFAULT_BUSY_TIMEOUT_MS, ensure_migrated
from .search_index import tokenize_for_index

VALID_TERMS_MODES = frozenset({"plaintext", "keyed"})

# Cap on how many freshly-encrypted rows are verified by round-trip decrypt
# before commit. Bounded so enable/rotate stay fast on large databases while
# still catching systematic bugs (wrong AAD, wrong key derivation, etc.).
MAX_VERIFY_SAMPLES = 5


class EncryptionLifecycleError(CryptoError):
    """Raised for invalid encryption state transitions.

    Examples: enabling when already enabled, rotating when not enabled, or
    operating on a database whose schema predates the encryption metadata
    table (migration v4).
    """


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {str(r[0]) for r in cur.fetchall()}


def _open_rw(db_path: Path, *, busy_timeout_ms: int) -> sqlite3.Connection:
    timeout_s = max(busy_timeout_ms / 1000.0, 0.001)
    conn = sqlite3.connect(db_path, timeout=timeout_s)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    return conn


@dataclass(slots=True)
class EncryptionStatus:
    """Read-only encryption diagnostics for one database (never mutates it)."""

    db_path: Path
    schema_ready: bool
    enabled: bool
    key_id: str | None
    algorithm: str | None
    envelope_version: int | None
    db_identity: str | None
    terms_mode: str
    updated_at: str | None
    cryptography_available: bool


def encryption_status(
    db_path: Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> EncryptionStatus:
    """Read the ``encryption_meta`` row without writing anything."""
    path = Path(db_path)
    not_ready = EncryptionStatus(
        db_path=path,
        schema_ready=False,
        enabled=False,
        key_id=None,
        algorithm=None,
        envelope_version=None,
        db_identity=None,
        terms_mode="plaintext",
        updated_at=None,
        cryptography_available=CRYPTOGRAPHY_AVAILABLE,
    )
    if not path.exists() or path.stat().st_size == 0:
        return not_ready

    from .storage import sqlite_readonly_uri

    conn = sqlite3.connect(sqlite_readonly_uri(path), uri=True, timeout=busy_timeout_ms / 1000.0)
    try:
        conn.execute("PRAGMA query_only = ON")
        if "encryption_meta" not in _table_names(conn):
            return not_ready
        row = conn.execute(
            """
            SELECT enabled, key_id, algorithm, envelope_version, db_identity, terms_mode, updated_at
            FROM encryption_meta WHERE id = 1
            """
        ).fetchone()
        if row is None:
            return not_ready
        return EncryptionStatus(
            db_path=path,
            schema_ready=True,
            enabled=bool(row[0]),
            key_id=row[1],
            algorithm=row[2],
            envelope_version=row[3],
            db_identity=row[4],
            terms_mode=str(row[5]),
            updated_at=row[6],
            cryptography_available=CRYPTOGRAPHY_AVAILABLE,
        )
    finally:
        conn.close()


@dataclass(slots=True)
class EncryptionOperationReport:
    """Result of :func:`enable_encryption` or :func:`rotate_encryption`."""

    db_path: Path
    action: str  # "enable" | "rotate"
    key_id: str
    previous_key_id: str | None
    db_identity: str
    terms_mode: str
    messages_encrypted: int = 0
    embeddings_encrypted: int = 0
    summaries_encrypted: int = 0
    agent_state_encrypted: int = 0
    terms_reindexed: int = 0
    verified_samples: int = 0
    backup_path: Path | None = None
    backup_contains_plaintext: bool = True
    vacuumed: bool = False
    message: str = "ok"


def _create_pre_operation_backup(db_path: Path, *, busy_timeout_ms: int, tag: str) -> Path:
    """Snapshot the (still plaintext) database before any encryption writes."""
    from .storage import sqlite_readonly_uri

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = db_path.with_name(f"{db_path.name}.pre-{tag}.{stamp}.bak")
    if candidate.exists():
        raise CryptoError(f"Backup path already exists: {candidate}")
    src = sqlite3.connect(sqlite_readonly_uri(db_path), uri=True, timeout=busy_timeout_ms / 1000.0)
    try:
        dest = sqlite3.connect(candidate)
        try:
            src.backup(dest)
            dest.commit()
        finally:
            dest.close()
    finally:
        src.close()
    return candidate


# Test-only failure injection: set to ``before_replace`` to fail after VACUUM INTO.
_INJECT_FAILURE_AT: str | None = None


def _maybe_inject(phase: str) -> None:
    if _INJECT_FAILURE_AT is not None and _INJECT_FAILURE_AT == phase:
        raise RuntimeError(f"injected failure at phase={phase}")


def _vacuum_into_and_replace(db_path: Path) -> None:
    """Rewrite the whole file via ``VACUUM INTO`` then atomically swap it in.

    Ensures freed pages that once held plaintext are not left behind in the
    live database file's free list. Must run on a *fresh* connection after
    the encrypting transaction has committed and its connection is closed.
    """
    tmp_path = db_path.with_name(db_path.name + ".vacuum-tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        mode = db_path.stat().st_mode & 0o777
    except OSError:
        mode = 0o600
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA wal_checkpoint(FULL)")
        conn.execute("VACUUM INTO ?", (str(tmp_path),))
    finally:
        conn.close()
    try:
        os.chmod(tmp_path, mode)
    except OSError:
        pass
    # Same-filesystem atomic replace after inject point for tests.
    _maybe_inject("before_replace")
    os.replace(tmp_path, db_path)
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass


def _reindex_terms_keyed(
    conn: sqlite3.Connection,
    ctx: EncryptionContext,
    message_id: int,
    plaintext_content: str,
) -> int:
    conn.execute("DELETE FROM memory_message_terms WHERE message_id = ?", (message_id,))
    terms = tokenize_for_index(plaintext_content)
    if not terms:
        return 0
    conn.executemany(
        "INSERT OR IGNORE INTO memory_message_terms (message_id, term) VALUES (?, ?)",
        [(message_id, keyed_term_token(ctx, term)) for term in terms],
    )
    return len(terms)


def _read_encryption_meta(conn: sqlite3.Connection) -> sqlite3.Row:
    if "encryption_meta" not in _table_names(conn):
        raise EncryptionLifecycleError(
            "Database schema does not have encryption support yet. Run migrations "
            "(schema v4+) before enabling encryption."
        )
    row = conn.execute(
        """
        SELECT enabled, key_id, algorithm, envelope_version, db_identity, terms_mode, updated_at
        FROM encryption_meta WHERE id = 1
        """
    ).fetchone()
    if row is None:
        raise EncryptionLifecycleError("encryption_meta singleton row is missing; database is corrupt.")
    return row


def _encrypt_all_rows(
    conn: sqlite3.Connection,
    ctx: EncryptionContext,
    *,
    terms_mode: str,
) -> tuple[int, int, int, int, int, list[tuple[int, str]]]:
    """Encrypt sensitive columns in place. Returns counts + verification samples.

    Must run inside the caller's write transaction. ``source`` rows are read
    as plaintext (either genuinely plaintext, or already-decrypted by the
    caller for rotation) immediately before being re-encrypted under *ctx*.
    """
    samples: list[tuple[int, str]] = []

    messages = conn.execute(
        "SELECT id, project, provider, conversation_id, role, content, metadata_json "
        "FROM memory_messages ORDER BY id ASC"
    ).fetchall()
    for row in messages:
        mid = int(row["id"])
        plaintext_content = str(row["content"])
        ct_content = encrypt_field(ctx, plaintext_content, table="memory_messages", column="content", row_identity=str(mid))
        ct_metadata = encrypt_field(
            ctx, str(row["metadata_json"]), table="memory_messages", column="metadata_json", row_identity=str(mid)
        )
        fingerprint = content_fingerprint(
            ctx,
            project=str(row["project"]),
            provider=str(row["provider"]),
            conversation_id=str(row["conversation_id"]),
            role=str(row["role"]),
            content=plaintext_content,
        )
        conn.execute(
            "UPDATE memory_messages SET content = ?, metadata_json = ?, content_fingerprint = ? WHERE id = ?",
            (ct_content, ct_metadata, fingerprint, mid),
        )
        if terms_mode == "keyed" and "memory_message_terms" in _table_names(conn):
            _reindex_terms_keyed(conn, ctx, mid, plaintext_content)
        if len(samples) < MAX_VERIFY_SAMPLES:
            samples.append((mid, plaintext_content))
    messages_encrypted = len(messages)

    embeddings = conn.execute("SELECT id, embedding_json FROM memory_embeddings ORDER BY id ASC").fetchall()
    for row in embeddings:
        eid = int(row["id"])
        ct = encrypt_field(
            ctx, str(row["embedding_json"]), table="memory_embeddings", column="embedding_json", row_identity=str(eid)
        )
        conn.execute("UPDATE memory_embeddings SET embedding_json = ? WHERE id = ?", (ct, eid))
    embeddings_encrypted = len(embeddings)

    summaries = conn.execute("SELECT id, summary FROM memory_summaries ORDER BY id ASC").fetchall()
    for row in summaries:
        sid = int(row["id"])
        ct = encrypt_field(ctx, str(row["summary"]), table="memory_summaries", column="summary", row_identity=str(sid))
        conn.execute("UPDATE memory_summaries SET summary = ? WHERE id = ?", (ct, sid))
    summaries_encrypted = len(summaries)

    agent_rows = conn.execute("SELECT id, value FROM agent_state ORDER BY id ASC").fetchall()
    for row in agent_rows:
        aid = int(row["id"])
        ct = encrypt_field(ctx, str(row["value"]), table="agent_state", column="value", row_identity=str(aid))
        conn.execute("UPDATE agent_state SET value = ? WHERE id = ?", (ct, aid))
    agent_state_encrypted = len(agent_rows)

    terms_reindexed = messages_encrypted if terms_mode == "keyed" else 0

    return (
        messages_encrypted,
        embeddings_encrypted,
        summaries_encrypted,
        agent_state_encrypted,
        terms_reindexed,
        samples,
    )


def _decrypt_all_rows_for_rotation(
    conn: sqlite3.Connection,
    ctx_old: EncryptionContext,
) -> None:
    """Decrypt every sensitive column back to plaintext, in place, using *ctx_old*.

    Intermediate step for rotation: run inside the same write transaction as
    :func:`_encrypt_all_rows` so a failure anywhere rolls back to the
    original (still-encrypted-under-the-old-key) state.
    """
    messages = conn.execute("SELECT id, content, metadata_json FROM memory_messages ORDER BY id ASC").fetchall()
    for row in messages:
        mid = int(row["id"])
        pt_content = decrypt_field(ctx_old, str(row["content"]), table="memory_messages", column="content", row_identity=str(mid))
        pt_metadata = decrypt_field(
            ctx_old, str(row["metadata_json"]), table="memory_messages", column="metadata_json", row_identity=str(mid)
        )
        conn.execute(
            "UPDATE memory_messages SET content = ?, metadata_json = ? WHERE id = ?",
            (pt_content, pt_metadata, mid),
        )

    embeddings = conn.execute("SELECT id, embedding_json FROM memory_embeddings ORDER BY id ASC").fetchall()
    for row in embeddings:
        eid = int(row["id"])
        pt = decrypt_field(
            ctx_old, str(row["embedding_json"]), table="memory_embeddings", column="embedding_json", row_identity=str(eid)
        )
        conn.execute("UPDATE memory_embeddings SET embedding_json = ? WHERE id = ?", (pt, eid))

    summaries = conn.execute("SELECT id, summary FROM memory_summaries ORDER BY id ASC").fetchall()
    for row in summaries:
        sid = int(row["id"])
        pt = decrypt_field(ctx_old, str(row["summary"]), table="memory_summaries", column="summary", row_identity=str(sid))
        conn.execute("UPDATE memory_summaries SET summary = ? WHERE id = ?", (pt, sid))

    agent_rows = conn.execute("SELECT id, value FROM agent_state ORDER BY id ASC").fetchall()
    for row in agent_rows:
        aid = int(row["id"])
        pt = decrypt_field(ctx_old, str(row["value"]), table="agent_state", column="value", row_identity=str(aid))
        conn.execute("UPDATE agent_state SET value = ? WHERE id = ?", (pt, aid))


def _verify_samples(
    conn: sqlite3.Connection,
    ctx: EncryptionContext,
    samples: list[tuple[int, str]],
) -> int:
    for mid, expected_plaintext in samples:
        row = conn.execute("SELECT content FROM memory_messages WHERE id = ?", (mid,)).fetchone()
        if row is None:
            raise CryptoError(f"Verification failed: message {mid} disappeared during encryption.")
        decrypted = decrypt_field(ctx, str(row[0]), table="memory_messages", column="content", row_identity=str(mid))
        if decrypted != expected_plaintext:
            raise CryptoError(f"Post-encryption verification mismatch for message {mid}.")
    return len(samples)


def enable_encryption(
    db_path: Path,
    *,
    master_key: bytes,
    terms_mode: str = "keyed",
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    create_backup: bool = True,
) -> EncryptionOperationReport:
    """Enable field-level encryption for *db_path* using *master_key*.

    On any failure the write transaction is rolled back and the database is
    left exactly as it was (still fully usable, still plaintext). The
    pre-encryption backup (if created) is never deleted by this function.
    """
    require_cryptography()
    if terms_mode not in VALID_TERMS_MODES:
        raise ValueError(f"terms_mode must be one of {sorted(VALID_TERMS_MODES)}, got {terms_mode!r}")

    path = Path(db_path)
    ensure_migrated(path, busy_timeout_ms=busy_timeout_ms)

    backup_path: Path | None = None
    if create_backup:
        backup_path = _create_pre_operation_backup(path, busy_timeout_ms=busy_timeout_ms, tag="encrypt")

    conn = _open_rw(path, busy_timeout_ms=busy_timeout_ms)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            meta = _read_encryption_meta(conn)
            if bool(meta["enabled"]):
                raise EncryptionLifecycleError(
                    "Encryption is already enabled for this database. Use rotate_encryption to change keys."
                )
            db_identity = str(meta["db_identity"])
            ctx = EncryptionContext(master_key, db_identity)

            (
                messages_encrypted,
                embeddings_encrypted,
                summaries_encrypted,
                agent_state_encrypted,
                terms_reindexed,
                samples,
            ) = _encrypt_all_rows(conn, ctx, terms_mode=terms_mode)

            verified = _verify_samples(conn, ctx, samples)

            conn.execute(
                """
                UPDATE encryption_meta
                SET enabled = 1, key_id = ?, algorithm = ?, envelope_version = ?,
                    terms_mode = ?, updated_at = ?
                WHERE id = 1
                """,
                (ctx.key_id, ALGORITHM, ENVELOPE_VERSION, terms_mode, _utc_now()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    vacuumed = False
    try:
        _vacuum_into_and_replace(path)
        vacuumed = True
    except (sqlite3.Error, OSError, RuntimeError):
        # Encryption already committed durably; scrubbing free pages is
        # best-effort and must not be reported as a failed enable.
        vacuumed = False
        tmp_path = path.with_name(path.name + ".vacuum-tmp")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return EncryptionOperationReport(
        db_path=path,
        action="enable",
        key_id=ctx.key_id,
        previous_key_id=None,
        db_identity=db_identity,
        terms_mode=terms_mode,
        messages_encrypted=messages_encrypted,
        embeddings_encrypted=embeddings_encrypted,
        summaries_encrypted=summaries_encrypted,
        agent_state_encrypted=agent_state_encrypted,
        terms_reindexed=terms_reindexed,
        verified_samples=verified,
        backup_path=backup_path,
        backup_contains_plaintext=True,
        vacuumed=vacuumed,
        message="ok",
    )


def rotate_encryption(
    db_path: Path,
    *,
    old_key: bytes,
    new_key: bytes,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    create_backup: bool = True,
) -> EncryptionOperationReport:
    """Re-encrypt an already-encrypted database under *new_key*.

    Raises :class:`~deepiri_memorymesh.crypto.WrongKeyError` immediately
    (before any writes) if *old_key* does not match the currently active
    key. Any failure during rotation rolls back to the original
    still-decryptable-with-``old_key`` state.
    """
    require_cryptography()
    path = Path(db_path)
    ensure_migrated(path, busy_timeout_ms=busy_timeout_ms)

    conn = _open_rw(path, busy_timeout_ms=busy_timeout_ms)
    backup_path: Path | None = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            meta = _read_encryption_meta(conn)
            if not bool(meta["enabled"]):
                raise EncryptionLifecycleError(
                    "Encryption is not enabled for this database. Use enable_encryption first."
                )
            db_identity = str(meta["db_identity"])
            terms_mode = str(meta["terms_mode"])
            previous_key_id = str(meta["key_id"])

            ctx_old = EncryptionContext(old_key, db_identity)
            if ctx_old.key_id != previous_key_id:
                raise WrongKeyError(
                    f"Provided old key (id {ctx_old.key_id!r}) does not match the active "
                    f"key (id {previous_key_id!r}); no changes were made."
                )
            ctx_new = EncryptionContext(new_key, db_identity)

            # Key validated while still holding the write lock (RESERVED, not
            # yet EXCLUSIVE) — a separate read-only connection can still
            # snapshot the pre-transaction state, mirroring migrations.py.
            if create_backup:
                backup_path = _create_pre_operation_backup(path, busy_timeout_ms=busy_timeout_ms, tag="rotate")

            _decrypt_all_rows_for_rotation(conn, ctx_old)
            (
                messages_encrypted,
                embeddings_encrypted,
                summaries_encrypted,
                agent_state_encrypted,
                terms_reindexed,
                samples,
            ) = _encrypt_all_rows(conn, ctx_new, terms_mode=terms_mode)

            verified = _verify_samples(conn, ctx_new, samples)

            conn.execute(
                """
                UPDATE encryption_meta
                SET key_id = ?, algorithm = ?, envelope_version = ?, updated_at = ?
                WHERE id = 1
                """,
                (ctx_new.key_id, ALGORITHM, ENVELOPE_VERSION, _utc_now()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    vacuumed = False
    try:
        _vacuum_into_and_replace(path)
        vacuumed = True
    except (sqlite3.Error, OSError, RuntimeError):
        vacuumed = False
        tmp_path = path.with_name(path.name + ".vacuum-tmp")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    return EncryptionOperationReport(
        db_path=path,
        action="rotate",
        key_id=ctx_new.key_id,
        previous_key_id=previous_key_id,
        db_identity=db_identity,
        terms_mode=terms_mode,
        messages_encrypted=messages_encrypted,
        embeddings_encrypted=embeddings_encrypted,
        summaries_encrypted=summaries_encrypted,
        agent_state_encrypted=agent_state_encrypted,
        terms_reindexed=terms_reindexed,
        verified_samples=verified,
        backup_path=backup_path,
        backup_contains_plaintext=False,
        vacuumed=vacuumed,
        message="ok",
    )
