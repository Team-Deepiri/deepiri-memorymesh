from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import MemoryRecord, now_iso
from .base import normalize_content, parse_generic_file, records_from_messages, safe_str


def _claude_messages_from_mapping(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    raw = mapping.get("chat_messages") or mapping.get("messages") or mapping.get("conversation")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for msg in raw:
        if not isinstance(msg, dict):
            continue
        role = safe_str(msg.get("role") or msg.get("sender") or msg.get("author"), "unknown")
        content = normalize_content(msg.get("content") or msg.get("text") or msg.get("parts"))
        ts = safe_str(msg.get("timestamp") or msg.get("created_at")) or now_iso()
        if not content:
            continue
        out.append(
            {
                "role": role,
                "content": content,
                "timestamp": ts,
                "metadata": {
                    "source": "claude",
                    "uuid": safe_str(msg.get("uuid")),
                },
            }
        )
    return out


def parse_claude_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    if file_path.suffix.lower() == ".jsonl":
        return parse_generic_file(provider, project, file_path)
    parsed = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(parsed, dict):
        conv_id = safe_str(
            parsed.get("conversation_id") or parsed.get("uuid") or parsed.get("id"),
            file_path.stem,
        )
        msgs = _claude_messages_from_mapping(parsed)
        if msgs:
            return records_from_messages(provider, project, conv_id, msgs)
    return parse_generic_file(provider, project, file_path)
