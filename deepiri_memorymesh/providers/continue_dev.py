from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import MemoryRecord, now_iso
from .base import normalize_content, parse_generic_file, records_from_messages, safe_str


def parse_continue_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    raw = file_path.read_text(encoding="utf-8")
    conv_id = file_path.stem
    messages: list[dict[str, Any]] = []

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
            role = safe_str(item.get("role") or item.get("author"), "unknown")
            content = normalize_content(item.get("content") or item.get("text") or item.get("message"))
            if not content:
                continue
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": safe_str(item.get("timestamp") or item.get("created_at")) or now_iso(),
                    "metadata": {"source": "continue-jsonl"},
                }
            )
        if messages:
            return records_from_messages(provider, project, conv_id, messages)
        return parse_generic_file(provider, project, file_path)

    parsed = json.loads(raw)
    rows: list[Any]
    if isinstance(parsed, dict):
        conv_id = safe_str(parsed.get("session_id") or parsed.get("conversation_id") or parsed.get("id"), conv_id)
        rows = parsed.get("messages") or parsed.get("history") or parsed.get("items") or []
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
                "metadata": {"source": "continue"},
            }
        )
    if messages:
        return records_from_messages(provider, project, conv_id, messages)
    return parse_generic_file(provider, project, file_path)
