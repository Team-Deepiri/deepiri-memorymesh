from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import MemoryRecord, now_iso
from .base import normalize_content, parse_generic_file, records_from_messages, safe_str


def _cursor_from_obj(data: dict[str, Any], file_path: Path) -> tuple[str, list[dict[str, Any]]]:
    conv_id = safe_str(
        data.get("conversationId") or data.get("conversation_id") or data.get("id"),
        file_path.stem,
    )
    rows = data.get("messages") or data.get("chat") or data.get("items") or []
    if not isinstance(rows, list):
        return conv_id, []
    msgs: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = safe_str(row.get("role") or row.get("type") or row.get("author"), "unknown")
        content = normalize_content(
            row.get("content") or row.get("text") or row.get("message") or row.get("parts")
        )
        if not content:
            continue
        ts = safe_str(row.get("timestamp") or row.get("createdAt") or row.get("created_at"))
        msgs.append(
            {
                "role": role,
                "content": content,
                "timestamp": ts or now_iso(),
                "metadata": {
                    "source": "cursor",
                    "toolCallId": safe_str(row.get("toolCallId") or row.get("tool_call_id")),
                },
            }
        )
    return conv_id, msgs


def parse_cursor_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    raw = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".jsonl":
        msgs: list[dict[str, Any]] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                continue
            role = safe_str(item.get("role") or item.get("type") or item.get("author"), "unknown")
            content = normalize_content(
                item.get("content") or item.get("text") or item.get("message") or item.get("parts")
            )
            if not content:
                continue
            msgs.append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": safe_str(item.get("timestamp") or item.get("createdAt")) or now_iso(),
                    "metadata": {"source": "cursor-jsonl"},
                }
            )
        return records_from_messages(provider, project, file_path.stem, msgs)

    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        conv_id, msgs = _cursor_from_obj(parsed, file_path)
        if msgs:
            return records_from_messages(provider, project, conv_id, msgs)
    return parse_generic_file(provider, project, file_path)
