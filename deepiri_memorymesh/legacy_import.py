"""Non-destructive importer for the historical simple-memory SQLite schema."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import default_db_path
from .embeddings import Embedder
from .models import MemoryRecord, now_iso
from .namespace import simple_api_ownership
from .storage import MemoryStore, detect_db_schema, sqlite_readonly_uri


def _legacy_default_source() -> Path:
    return Path.home() / ".memorymesh" / "memory.db"


LEGACY_DEFAULT_SOURCE = _legacy_default_source()

# Same complete ownership as Memory(project=...) for the chosen project.
LEGACY_PROVIDER = simple_api_ownership().provider
LEGACY_CONVERSATION_ID = simple_api_ownership().conversation_id
LEGACY_ROLE = simple_api_ownership().role

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@dataclass(slots=True)
class LegacyImportFailure:
    rowid: int | None
    error_type: str
    message: str


@dataclass(slots=True)
class LegacyImportReport:
    """Legacy import counts.

    - scanned: source rows examined
    - importable: valid rows eligible for a new insert (excludes duplicates)
    - imported: rows actually inserted (always 0 for dry-run)
    - duplicates_skipped: valid rows already present in the facade namespace
    - failed: malformed or failed rows
    """

    source: Path
    destination: Path
    project: str
    dry_run: bool
    scanned: int = 0
    importable: int = 0
    imported: int = 0
    duplicates_skipped: int = 0
    failed: int = 0
    failures: list[LegacyImportFailure] = field(default_factory=list)
    source_untouched: bool = True
    warning: str = (
        "Source database was not modified; it remains the caller's responsibility "
        "to retain or delete ~/.memorymesh/memory.db."
    )


def _paths_equivalent(a: Path, b: Path) -> bool:
    """True when two paths name the same filesystem object (incl. symlinks)."""
    try:
        if a.resolve() == b.resolve():
            return True
    except OSError:
        pass
    try:
        if a.exists() and b.exists() and a.samefile(b):
            return True
    except OSError:
        pass
    return False


def _validate_legacy_columns(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(memories)")
    cols = {str(row[1]) for row in cur.fetchall()}
    required = {"content", "embedding", "created_at"}
    missing = required - cols
    if missing:
        raise ValueError(
            f"Legacy table 'memories' is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )


def _canonical_timestamp(raw: object) -> tuple[str, str | None]:
    """Return ``(timestamp_for_row, original_if_not_used)``.

    Valid timezone-aware (or naive ISO) datetimes are preserved. UUID-shaped,
    missing, and malformed values are retained in metadata and replaced with a
    fresh canonical UTC timestamp.
    """
    if raw is None:
        return now_iso(), ""
    text = str(raw).strip()
    if not text:
        return now_iso(), ""
    if _UUID_RE.match(text):
        try:
            uuid.UUID(text)
        except ValueError:
            pass
        else:
            return now_iso(), text
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return now_iso(), text
    if parsed.tzinfo is None:
        # Preserve naive ISO strings as written historically.
        return text, None
    return text, None


def _existing_facade_message_id(
    dest: Path,
    ownership,
    content: str,
) -> int | None:
    if not dest.exists() or detect_db_schema(dest) != "platform":
        return None
    return MemoryStore(dest).find_message_id_by_exact_content(
        project=ownership.project,
        provider=ownership.provider,
        conversation_id=ownership.conversation_id,
        role=ownership.role,
        content=content,
    )


def import_legacy_memory(
    *,
    source: Path | None = None,
    destination: Path | None = None,
    project: str = "default",
    dry_run: bool = False,
    embedder: Embedder | None = None,
) -> LegacyImportReport:
    """Import rows from a legacy ``memories`` database into the platform schema.

    Opens *source* read-only. Never modifies *source*. Re-embeds with the active
    canonical embedder. Idempotent by exact content in the public Memory facade
    ownership namespace for *project*.
    """
    src = (source or _legacy_default_source()).expanduser()
    dest = (destination or default_db_path()).expanduser()
    ownership = simple_api_ownership(project)
    report = LegacyImportReport(
        source=src,
        destination=dest,
        project=ownership.project,
        dry_run=dry_run,
    )

    if not src.exists():
        raise FileNotFoundError(f"Legacy source database not found: {src}")

    if _paths_equivalent(src, dest):
        raise ValueError(
            f"Source and destination resolve to the same file: {src} / {dest}"
        )

    dest_kind = detect_db_schema(dest)
    if dest_kind == "legacy":
        raise ValueError(
            f"Destination {dest} contains the legacy-only schema and cannot be "
            "used as an import target. Choose a canonical/empty platform database."
        )
    if dest_kind in {"unknown", "corrupt"}:
        raise ValueError(
            f"Destination {dest} has schema kind {dest_kind!r} "
            "(ambiguous/unrelated/corrupt) and will not be written."
        )

    src_kind = detect_db_schema(src)
    if src_kind != "legacy":
        raise ValueError(
            f"Source {src} is not a legacy simple-memory database "
            f"(detected schema={src_kind!r}). Expected a SQLite file with the "
            "historical 'memories' table only."
        )

    source_bytes_before = src.read_bytes()
    dest_bytes_before = dest.read_bytes() if dest.exists() else None
    active = embedder or Embedder("fallback")
    store: MemoryStore | None = None
    if not dry_run:
        store = MemoryStore(dest)
        store.init()

    src_conn = sqlite3.connect(sqlite_readonly_uri(src), uri=True)
    src_conn.row_factory = sqlite3.Row
    try:
        _validate_legacy_columns(src_conn)
        rows = list(
            src_conn.execute(
                "SELECT rowid AS legacy_rowid, content, embedding, created_at "
                "FROM memories ORDER BY rowid ASC"
            )
        )
    finally:
        src_conn.close()

    report.scanned = len(rows)
    for row in rows:
        rowid = int(row["legacy_rowid"]) if row["legacy_rowid"] is not None else None
        try:
            content = row["content"]
            if content is None or not str(content):
                raise ValueError("empty content")
            content_s = str(content)
            # embedding column is ignored; we re-embed with the active backend.
            _ = row["embedding"]
            timestamp, original_ts = _canonical_timestamp(row["created_at"])
        except Exception as exc:
            report.failed += 1
            report.failures.append(
                LegacyImportFailure(
                    rowid=rowid,
                    error_type=type(exc).__name__,
                    message=str(exc) or repr(exc),
                )
            )
            continue

        if dry_run:
            if _existing_facade_message_id(dest, ownership, content_s) is not None:
                report.duplicates_skipped += 1
            else:
                report.importable += 1
            # imported stays 0 — nothing was written.
            continue

        assert store is not None
        existing = store.find_message_id_by_exact_content(
            project=ownership.project,
            provider=ownership.provider,
            conversation_id=ownership.conversation_id,
            role=ownership.role,
            content=content_s,
        )
        if existing is not None:
            report.duplicates_skipped += 1
            continue

        report.importable += 1
        meta: dict[str, object] = {
            "source": "legacy-simple-memory",
            "legacy_rowid": rowid,
        }
        if original_ts is None:
            meta["legacy_created_at"] = timestamp
        else:
            meta["legacy_created_at"] = original_ts
            meta["legacy_created_at_invalid"] = True
        rec = MemoryRecord(
            provider=ownership.provider,
            project=ownership.project,
            conversation_id=ownership.conversation_id,
            role=ownership.role,
            content=content_s,
            timestamp=timestamp,
            metadata_json=json.dumps(meta, ensure_ascii=True),
        )
        try:
            # Serialize lookup+insert+embed under the same IMMEDIATE transaction
            # used by the facade so concurrent importers cannot double-insert.
            embedding_json = active.dumps(active.embed(content_s))
            outcome, _msg_id = store.store_facade_memory_with_embedding(
                rec=rec,
                embedding_json=embedding_json,
            )
            if outcome == "duplicate":
                report.duplicates_skipped += 1
                # Was counted importable above; adjust if race lost the insert.
                report.importable = max(0, report.importable - 1)
            else:
                report.imported += 1
        except Exception as exc:
            report.failed += 1
            report.importable = max(0, report.importable - 1)
            report.failures.append(
                LegacyImportFailure(
                    rowid=rowid,
                    error_type=type(exc).__name__,
                    message=str(exc) or repr(exc),
                )
            )

    source_bytes_after = src.read_bytes()
    report.source_untouched = source_bytes_before == source_bytes_after
    if not report.source_untouched:
        raise RuntimeError(
            f"Legacy source database was modified during import: {src}. "
            "This is a bug; the source must remain untouched."
        )
    if dry_run:
        if dest_bytes_before is None:
            if dest.exists():
                raise RuntimeError(
                    f"Dry-run created destination database unexpectedly: {dest}"
                )
        elif dest.read_bytes() != dest_bytes_before:
            raise RuntimeError(
                f"Dry-run modified existing destination database: {dest}"
            )
    return report
