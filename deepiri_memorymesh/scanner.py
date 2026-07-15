"""Device-wide scan and ingest for Claude Code, Cursor, and OpenCode."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .device_paths import ProviderRoot, discover_provider_roots
from .models import MemoryRecord
from .providers import parse_provider_file
from .providers.cursor_sqlite import find_cursor_databases, parse_cursor_sqlite


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
