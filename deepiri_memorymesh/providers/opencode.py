from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import MemoryRecord, now_iso
from .base import normalize_content, parse_generic_file, records_from_messages, safe_str


def parse_opencode_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    raw = file_path.read_text(encoding="utf-8")
    messages: list[dict[str, Any]] = []
    conv_id = file_path.stem
    if file_path.suffix.lower() == ".jsonl":
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            evt = safe_str(item.get("type") or item.get("event"))
            payload = item.get("event") if isinstance(item.get("event"), dict) else item
            role = safe_str(payload.get("role") or payload.get("author") or payload.get("speaker"), "unknown")
            content = normalize_content(payload.get("content") or payload.get("text") or payload.get("message"))
            if evt.startswith("message") and content:
                messages.append(
                    {
                        "role": role,
                        "content": content,
                        "timestamp": safe_str(payload.get("timestamp")) or now_iso(),
                        "metadata": {"source": "opencode-event", "event_type": evt},
                    }
                )
        if messages:
            return records_from_messages(provider, project, conv_id, messages)
        return parse_generic_file(provider, project, file_path)

    parsed = json.loads(raw)
    rows: list[Any]
    if isinstance(parsed, dict):
        conv_id = safe_str(parsed.get("session_id") or parsed.get("conversation_id") or parsed.get("id"), conv_id)
        rows = parsed.get("messages") or parsed.get("events") or parsed.get("items") or []
    elif isinstance(parsed, list):
        rows = parsed
    else:
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = safe_str(row.get("role") or row.get("author"), "unknown")
        content = normalize_content(row.get("content") or row.get("text") or row.get("message"))
        if not content:
            continue
        messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": safe_str(row.get("timestamp") or row.get("created_at")) or now_iso(),
                "metadata": {"source": "opencode"},
            }
        )
    if messages:
        return records_from_messages(provider, project, conv_id, messages)
    return parse_generic_file(provider, project, file_path)
