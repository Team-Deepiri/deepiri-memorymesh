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


def _claude_session_line(item: dict[str, Any]) -> dict[str, Any] | None:
    """Parse one Claude Code session JSONL line (history or project transcript)."""
    display = normalize_content(item.get("display") or item.get("content"))
    msg_type = safe_str(item.get("type"))
    role = safe_str(item.get("role"))
    if not role:
        if msg_type in {"user", "human"}:
            role = "user"
        elif msg_type in {"assistant", "message"}:
            role = "assistant"
        elif display and item.get("sessionId"):
            role = "user"
        else:
            role = "unknown"
    content = display
    if not content:
        message = item.get("message")
        if isinstance(message, dict):
            content = normalize_content(message.get("content") or message.get("text"))
            role = safe_str(message.get("role"), role)
        elif isinstance(message, list):
            parts: list[str] = []
            for part in message:
                if isinstance(part, dict):
                    txt = normalize_content(part.get("text") or part.get("content"))
                    if txt:
                        parts.append(txt)
            content = "\n".join(parts)
    if not content:
        return None
    return {
        "role": role,
        "content": content,
        "timestamp": safe_str(item.get("timestamp") or item.get("created_at")) or now_iso(),
        "metadata": {
            "source": "claude-session",
            "session_id": safe_str(item.get("sessionId") or item.get("session_id")),
            "project_path": safe_str(item.get("project") or item.get("cwd")),
            "uuid": safe_str(item.get("uuid")),
        },
    }


def parse_claude_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    if file_path.suffix.lower() == ".jsonl":
        messages: list[dict[str, Any]] = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            parsed = _claude_session_line(item)
            if parsed:
                messages.append(parsed)
        if messages:
            return records_from_messages(provider, project, file_path.stem, messages)
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
