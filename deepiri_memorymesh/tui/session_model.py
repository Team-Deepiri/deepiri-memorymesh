"""Session browser model on top of the MemoryMesh SDK.

Pure, testable data layer for the interactive TUI. It knows nothing about
terminals: callers (``app.py``) feed in a :class:`.MemoryMesh` instance plus
optional on-disk sessions and get back plain rows and results. Disk sessions
come from :func:`deepiri_memorymesh.scanner.list_device_sessions`; database
conversations come from ``mesh.store.list_conversations``. Nothing here spawns
processes or starts detached services.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..providers import parse_generic_file, parse_provider_file
from ..providers.registry import get_provider
from ..scanner import DeviceSession
from ..sync_service import MemoryMesh, SyncAutoReport
from ..transfer_delivery import DeliveryResult


@dataclass(slots=True)
class SessionRow:
    """One session row in the machine-wide browser.

    ``project`` is only populated for database-backed rows; disk-only rows
    carry an empty project until they are ingested during transfer.
    """

    provider: str
    conversation_id: str
    source: str  # "db" | "disk" | "db+disk"
    project: str = ""
    workspace: str = ""
    path: Path | None = None
    mtime: float = 0.0
    message_count: int = 0
    last_user_preview: str = ""


@dataclass(slots=True)
class TransferOutcome:
    """Result of a TUI transfer action (bundle written + delivered to inbox)."""

    bundle_path: Path
    message_count: int
    delivery: DeliveryResult
    ingested_from_disk: bool = False


def _parse_timestamp(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _db_providers(mesh: MemoryMesh) -> list[str]:
    names: list[str] = []
    for name in mesh.settings.providers:
        key = name.strip().lower()
        if key and key not in names:
            names.append(key)
    return names


def build_session_list(
    mesh: MemoryMesh,
    disk_sessions: list[DeviceSession] | None = None,
    limit: int = 200,
) -> list[SessionRow]:
    """Merge database conversations (all projects) with on-disk sessions.

    One row per ``(provider, conversation_id)`` regardless of source. If the
    same session exists in the database *and* on disk, the row is marked
    ``db+disk`` and keeps the richer database metadata. Sorted newest-first.
    """
    mesh.init()
    rows: dict[tuple[str, str], SessionRow] = {}

    for project in mesh.list_projects():
        for prov in _db_providers(mesh):
            try:
                conversations = mesh.store.list_conversations(
                    project, provider=prov, limit=limit
                )
            except Exception:
                continue
            for conv in conversations or []:
                conv_id = str(conv.get("conversation_id") or "")
                if not conv_id:
                    continue
                key = (prov, conv_id)
                mtime = _parse_timestamp(str(conv.get("last_timestamp") or ""))
                candidate = SessionRow(
                    provider=prov,
                    conversation_id=conv_id,
                    source="db",
                    project=project,
                    mtime=mtime,
                    message_count=int(conv.get("message_count") or 0),
                    last_user_preview=str(conv.get("last_user_preview") or ""),
                )
                existing = rows.get(key)
                if existing is None or candidate.message_count > existing.message_count:
                    rows[key] = candidate

    for disk in disk_sessions or []:
        key = (disk.provider, disk.conversation_id)
        existing = rows.get(key)
        if existing is None:
            rows[key] = SessionRow(
                provider=disk.provider,
                conversation_id=disk.conversation_id,
                source="disk",
                workspace=disk.workspace,
                path=disk.path,
                mtime=disk.mtime,
                last_user_preview=disk.preview,
            )
        else:
            existing.source = "db+disk"
            existing.workspace = disk.workspace
            existing.path = disk.path
            existing.mtime = max(existing.mtime, disk.mtime)
            if not existing.last_user_preview:
                existing.last_user_preview = disk.preview

    return sorted(rows.values(), key=lambda r: r.mtime, reverse=True)


def filter_sessions(
    rows: list[SessionRow],
    provider: str | None = None,
    text: str | None = None,
) -> list[SessionRow]:
    """Filter rows by provider key and/or free-text substring (case-insensitive)."""
    needle = (text or "").strip().lower()
    out: list[SessionRow] = []
    for row in rows:
        if provider:
            key = provider.strip().lower()
            if key != row.provider and key not in (get_provider(row.provider).aliases or ()):
                continue
        if needle:
            haystack = " ".join(
                [
                    row.provider,
                    row.conversation_id,
                    row.workspace,
                    row.project,
                    row.last_user_preview,
                ]
            ).lower()
            if needle not in haystack:
                continue
        out.append(row)
    return out


def conversation_messages(mesh: MemoryMesh, row: SessionRow) -> list[dict]:
    """Return message dicts for a session row (DB-first, else parse the disk file)."""
    mesh.init()
    if "db" in row.source and row.project:
        return list(
            mesh.store.list_messages_for_namespace(
                project=row.project,
                provider=row.provider,
                conversation_id=row.conversation_id,
            )
        )
    if row.path is not None and row.path.is_file():
        try:
            records = parse_provider_file(
                row.provider, row.project or "default", row.path
            )
        except Exception:
            records = parse_generic_file(
                row.provider, row.project or "default", row.path
            )
        return [
            {
                "role": rec.role,
                "content": rec.content,
                "timestamp": rec.timestamp,
                "conversation_id": rec.conversation_id,
            }
            for rec in records
        ]
    return []


def conversation_summary(mesh: MemoryMesh, row: SessionRow) -> str:
    """Return the stored compressed summary for a DB-backed session, if any."""
    if not row.project:
        return ""
    try:
        for summary in mesh.store.list_summaries(row.project):
            conv_id = str(summary.get("conversation_id") or "")
            if conv_id == row.conversation_id or conv_id in row.conversation_id:
                return str(summary.get("summary") or "")
    except Exception:
        return ""
    return ""


def transfer_destinations(mesh: MemoryMesh, source: str) -> list[str]:
    """Configured native providers other than *source* (sampled from settings)."""
    source = source.strip().lower()
    out: list[str] = []
    for name in mesh.settings.providers:
        key = name.strip().lower()
        if not key or key == source:
            continue
        cap = get_provider(key)
        if cap is not None and cap.parser_kind == "native":
            if key not in out:
                out.append(key)
    return sorted(out)


def transfer_session(
    mesh: MemoryMesh,
    row: SessionRow,
    to_provider: str,
    *,
    project: str | None = None,
    sync_source: bool = False,
    compress_first: bool = False,
    copy_clipboard: bool = False,
) -> TransferOutcome:
    """Transfer a session into another provider's inbox.

    Disk-only rows are ingested into the target project first (reusing the
    existing ``ingest_file`` SDK path) so the transaction stays on top of the
    established read path. Returns the transferred bundle plus delivery info.
    """
    target = to_provider.strip().lower()
    target_project = project or row.project or "default"
    mesh.init()

    ingested_from_disk = False
    if row.source == "disk" and row.path is not None and row.path.is_file():
        mesh.ingest_file(
            provider=row.provider, project=target_project, file_path=row.path
        )
        ingested_from_disk = True
    elif sync_source and row.path is not None and row.path.is_file():
        mesh.sync_directory_report(
            provider=row.provider,
            project=target_project,
            directory=row.path.parent,
            include_globs=[row.path.name],
        )

    if compress_first:
        mesh.compress_project(target_project)

    bundle_path, message_count, _push = mesh.transfer_with_report(
        project=target_project,
        from_provider=row.provider,
        to_provider=target,
        include_summaries=True,
        conversation_id=row.conversation_id,
    )
    delivery = mesh.deliver_transfer(bundle_path, target)
    if copy_clipboard:
        from ..transfer_delivery import try_clipboard_copy

        try_clipboard_copy(delivery.context_md.read_text(encoding="utf-8"))
    return TransferOutcome(
        bundle_path=bundle_path,
        message_count=message_count,
        delivery=delivery,
        ingested_from_disk=ingested_from_disk,
    )


def export_conversation(
    mesh: MemoryMesh,
    row: SessionRow,
    fmt: str = "md",
) -> tuple[str, str]:
    """Render one conversation as text (md or json). Returns (content, slug)."""
    messages = conversation_messages(mesh, row)
    slug = f"{row.provider}-{row.conversation_id}"
    if fmt.lower() == "json":
        import json

        payload = {
            "provider": row.provider,
            "conversation_id": row.conversation_id,
            "messages": [
                {
                    "role": m.get("role", ""),
                    "content": m.get("content", ""),
                    "timestamp": m.get("timestamp", ""),
                }
                for m in messages
            ],
        }
        return json.dumps(payload, ensure_ascii=True, indent=2), slug
    lines = [f"# {row.provider} session {row.conversation_id}", ""]
    for m in messages:
        role = m.get("role", "unknown")
        body = str(m.get("content", "")).strip()
        lines.append(f"## {role}")
        lines.append(body or "(empty)")
        lines.append("")
    return "\n".join(lines), slug


def project_stats(mesh: MemoryMesh, project: str) -> dict:
    mesh.init()
    return mesh.stats(project)


def compress_project(mesh: MemoryMesh, project: str) -> int:
    mesh.init()
    return mesh.compress_project(project)


def embed_project(mesh: MemoryMesh, project: str) -> int:
    mesh.init()
    return mesh.embed_project(project)


def scan_and_ingest(
    mesh: MemoryMesh, project: str, providers: list[str] | None = None
) -> object:
    mesh.init()
    return mesh.ingest_device(project=project, providers=providers)


def sync_all(mesh: MemoryMesh, project: str) -> SyncAutoReport:
    mesh.init()
    return mesh.sync_auto_report(project)