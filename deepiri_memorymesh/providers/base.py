from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import MemoryRecord, now_iso


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _plain_text_messages(raw: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    _ROLE_PREFIXES = (
        ("user:", "user"),
        ("human:", "user"),
        ("you:", "user"),
        ("assistant:", "assistant"),
        ("ai:", "assistant"),
        ("bot:", "assistant"),
        ("system:", "system"),
        ("hint:", "system"),
    )
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        role = "assistant"
        body = stripped
        for prefix, mapped in _ROLE_PREFIXES:
            if lower.startswith(prefix):
                role = mapped
                body = stripped.split(":", 1)[1].strip()
                break
        messages.append({"role": role, "content": body})
    return messages


def normalize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") or item.get("content") or item.get("value")
                if txt:
                    chunks.append(safe_str(txt))
        return "\n".join(c for c in chunks if c).strip()
    if isinstance(value, dict):
        txt = value.get("text") or value.get("content") or value.get("value")
        return safe_str(txt)
    return safe_str(value)


def records_from_messages(
    provider: str,
    project: str,
    conversation_id: str,
    messages: list[dict[str, Any]],
) -> list[MemoryRecord]:
    out: list[MemoryRecord] = []
    for msg in messages:
        role = safe_str(msg.get("role") or msg.get("author") or msg.get("speaker"), "unknown")
        content = normalize_content(
            msg.get("content") or msg.get("text") or msg.get("message") or msg.get("parts")
        )
        if not content:
            continue
        timestamp = safe_str(
            msg.get("timestamp")
            or msg.get("created_at")
            or msg.get("time")
            or msg.get("date")
        ) or now_iso()
        metadata = msg.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}
        metadata_json = json.dumps(metadata, ensure_ascii=True)
        out.append(
            MemoryRecord(
                provider=provider,
                project=project,
                conversation_id=conversation_id,
                role=role,
                content=content,
                timestamp=timestamp,
                metadata_json=metadata_json,
            )
        )
    return out


def parse_generic_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    """
    Expected flexible input format:
      - JSON object with:
          conversation_id: str
          messages: [{role, content, timestamp?, metadata?}, ...]
      - JSONL where each line is one message object
    """
    raw = file_path.read_text(encoding="utf-8")
    provider_normalized = provider.strip().lower()
    if file_path.suffix.lower() == ".jsonl":
        conv_id = file_path.stem
        messages: list[dict[str, Any]] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                messages.append(item)
        return records_from_messages(provider_normalized, project, conv_id, messages)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Plain-text transcripts (e.g. Cursor agent-transcripts/*.txt) are a
        # legitimate input format; parse them line-by-line instead of failing.
        return records_from_messages(
            provider_normalized, project, file_path.stem, _plain_text_messages(raw)
        )
    if isinstance(parsed, list):
        conv_id = file_path.stem
        messages = parsed
    else:
        conv_id = safe_str(
            parsed.get("conversation_id")
            or parsed.get("id")
            or parsed.get("chat_id")
            or parsed.get("session_id"),
            file_path.stem,
        )
        messages = (
            parsed.get("messages")
            or parsed.get("conversation")
            or parsed.get("items")
            or parsed.get("turns")
            or []
        )

    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    return records_from_messages(provider_normalized, project, conv_id, messages)


def parse_provider_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    return parse_generic_file(provider, project, file_path)
