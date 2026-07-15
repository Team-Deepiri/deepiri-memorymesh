"""Portable u-data packages for cross-provider memory transfer."""

from __future__ import annotations

import json
import platform
import socket
import tarfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .device_paths import discover_provider_roots
from .models import MemoryRecord, now_iso
from .scanner import DeviceScanReport, ingest_device, scan_device


FORMAT_VERSION = "memorymesh-u-data-v1"


@dataclass(slots=True)
class UDataManifest:
    format: str = FORMAT_VERSION
    created_at: str = field(default_factory=now_iso)
    hostname: str = field(default_factory=socket.gethostname)
    platform: str = field(default_factory=platform.platform)
    project: str = ""
    providers: dict[str, dict[str, int]] = field(default_factory=dict)
    source_locations: list[dict[str, str]] = field(default_factory=list)
    message_count: int = 0
    summary_count: int = 0


def build_udata_payload(
    project: str,
    messages: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    scan: DeviceScanReport | None = None,
) -> dict[str, Any]:
    manifest = UDataManifest(project=project, message_count=len(messages), summary_count=len(summaries))
    if scan:
        for loc in scan.locations:
            if loc.exists:
                manifest.source_locations.append(
                    {
                        "provider": loc.provider,
                        "path": str(loc.path),
                        "kind": loc.kind,
                        "description": loc.description,
                    }
                )
        for res in scan.results:
            manifest.providers.setdefault(res.provider, {"files": 0, "messages": 0})
            manifest.providers[res.provider]["files"] += res.files_found
            manifest.providers[res.provider]["messages"] += res.messages_ingested

    return {
        "manifest": asdict(manifest),
        "messages": messages,
        "summaries": summaries,
    }


def write_udata_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.expanduser()
    if output_path.suffix != ".json":
        output_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path


def write_udata_archive(payload: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.expanduser()
    name = output_path.name
    if not (name.endswith(".tar.gz") or name.endswith(".tgz")):
        output_path = output_path.parent / f"{output_path.name}.tar.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        json_path = tmp_path / "udata.json"
        write_udata_json(payload, json_path)
        readme = tmp_path / "README-udata.txt"
        readme.write_text(
            "MemoryMesh u-data package.\n"
            "Import: memorymesh package import ./udata.tar.gz -p YOUR_PROJECT\n",
            encoding="utf-8",
        )
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(json_path, arcname="udata.json")
            tar.add(readme, arcname="README-udata.txt")
    return output_path


def load_udata(path: Path) -> dict[str, Any]:
    path = path.expanduser()
    if path.suffix in {".gz", ".tgz"} or str(path).endswith(".tar.gz"):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(path, "r:gz") as tar:
                tar.extractall(tmp)
            extracted = Path(tmp) / "udata.json"
            if not extracted.exists():
                for candidate in Path(tmp).rglob("*.json"):
                    extracted = candidate
                    break
            return json.loads(extracted.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def udata_to_memory_records(
    payload: dict[str, Any],
    project_override: str | None = None,
) -> list[MemoryRecord]:
    project = project_override or str(payload.get("manifest", {}).get("project") or "default")
    records: list[MemoryRecord] = []
    for msg in payload.get("messages") or []:
        records.append(
            MemoryRecord(
                provider=str(msg.get("provider") or "udata"),
                project=project,
                conversation_id=str(msg.get("conversation_id") or "imported"),
                role=str(msg.get("role") or "unknown"),
                content=str(msg.get("content") or ""),
                timestamp=str(msg.get("timestamp") or now_iso()),
                metadata_json=str(msg.get("metadata_json") or "{}"),
            )
        )
    return records


def transfer_payload_for_provider(
    messages: list[dict[str, Any]],
    project: str,
    from_provider: str,
    to_provider: str,
) -> dict[str, Any]:
    """Build provider-import-friendly transfer JSON."""
    filtered = [m for m in messages if str(m.get("provider", "")).lower() == from_provider.lower()]
    return {
        "format": "memorymesh-transfer-v1",
        "project": project,
        "from_provider": from_provider,
        "to_provider": to_provider,
        "conversation_id": f"transfer-{from_provider}-to-{to_provider}",
        "messages": [
            {
                "role": m.get("role"),
                "content": m.get("content"),
                "timestamp": m.get("timestamp"),
                "metadata": {
                    "origin_provider": m.get("provider"),
                    "origin_conversation_id": m.get("conversation_id"),
                    "transfer": True,
                },
            }
            for m in filtered
        ],
    }
