"""Device-wide scan and ingest for Claude Code, Cursor, and OpenCode."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .device_paths import ProviderRoot, discover_provider_roots
from .models import MemoryRecord
from .providers import parse_provider_file
from .providers.cursor_sqlite import find_cursor_databases, parse_cursor_sqlite
from .session_bridge import preview_from_record


@dataclass(slots=True)
class ScanResult:
    provider: str
    path: Path
    kind: str
    exists: bool
    files_found: int = 0
    messages_ingested: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DeviceScanReport:
    locations: list[ProviderRoot] = field(default_factory=list)
    results: list[ScanResult] = field(default_factory=list)
    total_messages: int = 0

    def summary_lines(self) -> list[str]:
        lines = ["Device scan summary:", ""]
        for loc in self.locations:
            status = "found" if loc.exists else "missing"
            lines.append(f"  [{status}] {loc.provider:10} {loc.path}")
            lines.append(f"           {loc.description}")
            if loc.exists:
                lines.append(f"           files≈{loc.file_count}")
        lines.append("")
        for res in self.results:
            if res.messages_ingested:
                lines.append(
                    f"  ingested {res.messages_ingested:5} msgs  {res.provider:10} {res.path}"
                )
        lines.append(f"\nTotal messages ingested: {self.total_messages}")
        return lines


@dataclass(slots=True)
class DeviceSession:
    """One on-disk provider session transcript found by a device scan.

    Provides a lightweight, non-destructive view of local sessions before (or
    without) ingesting anything into the memory database.
    """

    provider: str
    conversation_id: str
    path: Path
    workspace: str
    mtime: float
    preview: str = ""


_PROJECT_DIR_MARKERS = ("projects", "project")


def _workspace_label(path: Path) -> str:
    """Derive a human workspace label from a provider session file path."""
    parts = path.parts
    for idx, part in enumerate(parts):
        if part in _PROJECT_DIR_MARKERS and idx + 1 < len(parts):
            return parts[idx + 1].lstrip("-")
    return path.parent.name.lstrip("-") or "unknown"


def _first_record_from_json(data: object) -> dict | None:
    if isinstance(data, dict):
        for key in ("messages", "events", "items", "chat_messages", "conversation"):
            val = data.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        return item
                return None
        return data if data else None
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                return item
    return None


def _session_preview(path: Path) -> str:
    """Short human preview of a session transcript file (never loads whole files).

    JSON/JSONL files are accepted only when their first record looks like a
    chat message (display/content/text/message keys). Everything else returns
    the raw head of the file. Returns ``""`` for files with no readable
    conversation content so callers can skip noise (diffs, snapshots, etc.).
    """
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        try:
            if suffix == ".jsonl":
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(item, dict):
                            return (preview_from_record(item) or "")[:160]
                return ""
            if path.stat().st_size >= 4 * 1024 * 1024:
                return ""
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return ""
        rec = _first_record_from_json(data)
        if rec is not None:
            return (preview_from_record(rec) or "")[:160]
        return ""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")[:200]
        return raw.replace("\n", " ").strip()
    except OSError:
        return ""


def list_device_sessions() -> list[DeviceSession]:
    """List on-disk provider session transcripts machine-wide.

    Walks the same provider roots as :func:`ingest_device` but is strictly
    read-only: nothing is parsed into memory records and nothing is written.
    File-based roots only — Cursor SQLite chat databases surface through the
    database conversation list once ingested. Results are deduplicated by
    resolved path and sorted newest-first.
    """
    locations = discover_provider_roots()
    found: dict[Path, DeviceSession] = {}

    def _add_session(provider: str, file_path: Path) -> None:
        if not file_path.is_file():
            return
        if file_path.name == "history.jsonl":
            return
        try:
            key = file_path.resolve(strict=False)
        except OSError:
            key = file_path
        if key in found:
            return
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        preview = _session_preview(file_path)
        if not preview:
            return
        found[key] = DeviceSession(
            provider=provider,
            conversation_id=file_path.stem,
            path=file_path,
            workspace=_workspace_label(file_path),
            mtime=mtime,
            preview=preview,
        )

    for loc in locations:
        if not loc.exists:
            continue
        if loc.kind in {"sqlite", "sqlite_tree"}:
            continue
        if loc.path.is_file():
            _add_session(loc.provider, loc.path)
            continue
        if not loc.path.is_dir() or not loc.globs:
            continue
        for pattern in loc.globs:
            norm = pattern[3:] if pattern.startswith("**/") else pattern
            for fp in loc.path.rglob(norm):
                _add_session(loc.provider, fp)

    sessions = list(found.values())
    sessions.sort(key=lambda s: s.mtime, reverse=True)
    return sessions


def scan_device() -> DeviceScanReport:
    """Discover all configured provider data locations on this machine."""
    locations = discover_provider_roots()
    return DeviceScanReport(locations=locations)


def _workspace_hint_from_json(workspace_dir: Path) -> str:
    wj = workspace_dir / "workspace.json"
    if not wj.exists():
        return workspace_dir.name[:12]
    try:
        data = json.loads(wj.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return workspace_dir.name[:12]
    folder = data.get("folder") or data.get("workspace") or ""
    if isinstance(folder, str) and folder:
        return Path(folder.replace("file://", "")).name or workspace_dir.name[:12]
    return workspace_dir.name[:12]


def ingest_device(
    project: str,
    store=None,
    providers: list[str] | None = None,
) -> DeviceScanReport:
    """Scan device and ingest Claude / Cursor / OpenCode data into memory."""
    report = scan_device()
    allowed = {p.strip().lower() for p in (providers or ["claude", "cursor", "opencode"])}

    def _save(records: list[MemoryRecord]) -> int:
        if store is not None:
            return store.insert_messages(records)
        return len(records)

    cursor_dbs_done: set[str] = set()

    for loc in report.locations:
        if loc.provider not in allowed or not loc.exists:
            continue
        result = ScanResult(
            provider=loc.provider,
            path=loc.path,
            kind=loc.kind,
            exists=True,
        )

        try:
            if loc.provider == "cursor" and loc.kind in {"sqlite", "sqlite_tree"}:
                if cursor_dbs_done:
                    continue
                for db in find_cursor_databases():
                    key = str(db.resolve())
                    if key in cursor_dbs_done:
                        continue
                    cursor_dbs_done.add(key)
                    ws_hint = ""
                    if "workspaceStorage" in str(db):
                        ws_hint = _workspace_hint_from_json(db.parent)
                    recs = parse_cursor_sqlite("cursor", project, db, workspace_hint=ws_hint)
                    result.files_found += 1
                    n = _save(recs) if recs else 0
                    result.messages_ingested += n
                    report.total_messages += n
                if result.files_found or result.messages_ingested:
                    report.results.append(result)
                continue
            elif loc.path.is_file():
                recs = parse_provider_file(loc.provider, project, loc.path)
                result.files_found = 1
                n = _save(recs) if recs else 0
                result.messages_ingested = n
                report.total_messages += n
            elif loc.path.is_dir() and loc.globs:
                for pattern in loc.globs:
                    norm = pattern[3:] if pattern.startswith("**/") else pattern
                    for fp in loc.path.rglob(norm):
                        if not fp.is_file():
                            continue
                        if fp.suffix.lower() not in {".json", ".jsonl", ".txt", ".md"}:
                            continue
                        try:
                            recs = parse_provider_file(loc.provider, project, fp)
                            result.files_found += 1
                            n = _save(recs) if recs else 0
                            result.messages_ingested += n
                            report.total_messages += n
                        except Exception as exc:
                            result.errors.append(f"{fp}: {exc}")
        except Exception as exc:
            result.errors.append(str(exc))

        if result.files_found or result.messages_ingested or result.errors:
            report.results.append(result)

    return report
