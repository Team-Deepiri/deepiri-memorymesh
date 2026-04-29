from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import MemoryRecord, now_iso
from .base import normalize_content, parse_generic_file, records_from_messages, safe_str


def parse_gemini_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    if file_path.suffix.lower() == ".jsonl":
        return parse_generic_file(provider, project, file_path)
    parsed = json.loads(file_path.read_text(encoding="utf-8"))
    conv_id = file_path.stem
    messages: list[dict[str, Any]] = []

    if isinstance(parsed, dict):
        conv_id = safe_str(parsed.get("conversation_id") or parsed.get("session_id") or parsed.get("id"), conv_id)
        turns = parsed.get("messages") or parsed.get("turns") or parsed.get("events") or []
    elif isinstance(parsed, list):
        turns = parsed
    else:
        turns = []

    for item in turns:
        if not isinstance(item, dict):
            continue
        role = safe_str(item.get("role") or item.get("author") or item.get("speaker"), "unknown")
        content = normalize_content(item.get("content") or item.get("text") or item.get("parts") or item.get("message"))
        if not content:
            continue
        messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": safe_str(item.get("timestamp") or item.get("created_at") or item.get("time")) or now_iso(),
                "metadata": {"source": "gemini"},
            }
        )
    if messages:
        return records_from_messages(provider, project, conv_id, messages)
    return parse_generic_file(provider, project, file_path)
