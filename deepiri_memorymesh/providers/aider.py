"""Aider history parsers with stable identities for re-ingestion (T20).

Primary format: ``.aider.chat.history.md`` as written by Aider's InputOutput
(``####`` user turns, plain assistant text, ``>`` tool/status lines, and
``# aider chat started at …`` session headers). Parsing mirrors Aider's
``split_chat_history_markdown`` markers, with fenced-code awareness so
heading-like lines inside fences do not split turns.

Secondary: ``.aider.input.history`` (prompt_toolkit FileHistory) is treated as
user-input only — no fabricated assistant turns.

Timestamps never use mutable filesystem mtime for historical turns: session
headers when present, otherwise a fixed synthetic epoch plus turn ordinal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from ..models import MemoryRecord

logger = logging.getLogger(__name__)

_SESSION_STARTED_RE = re.compile(
    r"^#\s+aider chat started at\s+(.+?)\s*$",
    re.IGNORECASE,
)
_ISO_FRAG_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)

# Deterministic base when no evidenced Aider session timestamp exists.
# Independent of mutable file mtime so append/reparse cannot shift prior turns.
SYNTHETIC_EPOCH_ISO = "1970-01-01T00:00:00+00:00"
TIMESTAMP_ORIGIN_SESSION = "aider_session_header"
TIMESTAMP_ORIGIN_SYNTHETIC = "synthetic_epoch"
TIMESTAMP_ORIGIN_INPUT_COMMENT = "input_history_comment"


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def conversation_id_for_path(file_path: Path) -> str:
    """Deterministic conversation id from the history/session source path."""
    try:
        key = str(file_path.expanduser().resolve())
    except OSError:
        key = str(file_path.expanduser())
    return f"aider:{_sha256_hex(key)[:24]}"


def source_key_for_turn(
    *,
    conversation_id: str,
    turn_ordinal: int,
    role: str,
    content: str,
) -> str:
    """Deterministic per-turn source key (not content-only)."""
    digest = _sha256_hex(content)
    return f"{conversation_id}:{turn_ordinal}:{role}:{digest[:16]}"


def _parse_session_timestamp(header_value: str) -> str | None:
    raw = header_value.strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    m = _ISO_FRAG_RE.search(raw)
    if not m:
        return None
    frag = m.group(1).replace(" ", "T")
    try:
        if frag.endswith("Z"):
            frag = frag[:-1] + "+00:00"
        dt = datetime.fromisoformat(frag)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def _stable_turn_timestamp(
    *,
    base_iso: str,
    turn_ordinal: int,
    origin: str,
) -> tuple[str, dict]:
    """Combine evidenced/synthetic base with turn order (never wall-clock/mtime)."""
    try:
        base = datetime.fromisoformat(base_iso)
    except ValueError:
        base = datetime(1970, 1, 1, tzinfo=timezone.utc)
        origin = TIMESTAMP_ORIGIN_SYNTHETIC
        base_iso = SYNTHETIC_EPOCH_ISO
    stamped = base.replace(microsecond=min(turn_ordinal, 999999))
    synthetic = origin == TIMESTAMP_ORIGIN_SYNTHETIC
    meta = {
        "timestamp_origin": origin,
        "timestamp_base": base_iso,
        "timestamp_synthetic": synthetic,
        "turn_ordinal": turn_ordinal,
    }
    return stamped.isoformat(), meta


def _make_record(
    *,
    role: str,
    chunks: list[str],
    provider: str,
    project: str,
    conversation_id: str,
    turn_ordinal: int,
    base_iso: str,
    base_origin: str,
    source_filename: str,
    header_text: str | None,
) -> MemoryRecord | None:
    content = "".join(chunks).strip("\n")
    if not content.strip():
        return None
    role_norm = {"user": "user", "assistant": "assistant", "tool": "system"}.get(role, role)
    meta: dict = {
        "source": "aider-chat-history",
        "source_filename": source_filename,
        "turn_ordinal": turn_ordinal,
        "recognized_header": header_text,
        "provider_role": role,
    }
    ts, ts_meta = _stable_turn_timestamp(
        base_iso=base_iso, turn_ordinal=turn_ordinal, origin=base_origin
    )
    meta.update(ts_meta)
    sk = source_key_for_turn(
        conversation_id=conversation_id,
        turn_ordinal=turn_ordinal,
        role=role_norm,
        content=content,
    )
    return MemoryRecord(
        provider=provider,
        project=project,
        conversation_id=conversation_id,
        role=role_norm,
        content=content,
        timestamp=ts,
        metadata_json=json.dumps(meta, ensure_ascii=True),
        source_key=sk,
    )


def parse_aider_chat_history(
    provider: str,
    project: str,
    file_path: Path,
    *,
    text: str | None = None,
) -> tuple[list[MemoryRecord], list[str]]:
    """Parse ``.aider.chat.history.md``-style markdown."""
    diagnostics: list[str] = []
    if text is None:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            diagnostics.append(f"read_error:{type(exc).__name__}")
            return [], diagnostics

    conversation_id = conversation_id_for_path(file_path)
    source_filename = file_path.name
    # Default: fixed synthetic epoch (not mutable mtime).
    base_iso = SYNTHETIC_EPOCH_ISO
    base_origin = TIMESTAMP_ORIGIN_SYNTHETIC

    records: list[MemoryRecord] = []
    user: list[str] = []
    assistant: list[str] = []
    tool: list[str] = []
    turn_ordinal = 0
    current_user_header: str | None = None
    in_fence = False

    def append_msg(role: str, lines: list[str], *, header: str | None = None) -> None:
        nonlocal turn_ordinal
        if not "".join(lines).strip():
            return
        turn_ordinal += 1
        rec = _make_record(
            role=role,
            chunks=lines,
            provider=provider,
            project=project,
            conversation_id=conversation_id,
            turn_ordinal=turn_ordinal,
            base_iso=base_iso,
            base_origin=base_origin,
            source_filename=source_filename,
            header_text=header,
        )
        if rec is not None:
            records.append(rec)

    def active_append(line: str) -> None:
        if user:
            user.append(line)
        elif tool:
            tool.append(line)
        else:
            assistant.append(line)

    for line_no, line in enumerate(text.splitlines(keepends=True), start=1):
        try:
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                active_append(line)
                continue
            if in_fence:
                active_append(line)
                continue
            if line.startswith("# "):
                m = _SESSION_STARTED_RE.match(line.rstrip("\n"))
                if m:
                    parsed = _parse_session_timestamp(m.group(1))
                    if parsed:
                        base_iso = parsed
                        base_origin = TIMESTAMP_ORIGIN_SESSION
                continue
            if line.startswith("> "):
                append_msg("assistant", assistant)
                assistant = []
                append_msg("user", user, header=current_user_header)
                user = []
                current_user_header = None
                tool.append(line[2:])
                continue
            if line.startswith("#### "):
                append_msg("assistant", assistant)
                assistant = []
                append_msg("tool", tool)
                tool = []
                current_user_header = "####"
                user.append(line[5:])
                continue

            append_msg("user", user, header=current_user_header)
            user = []
            current_user_header = None
            append_msg("tool", tool)
            tool = []
            assistant.append(line)
        except Exception as exc:
            diagnostics.append(f"line_{line_no}:{type(exc).__name__}")
            continue

    append_msg("assistant", assistant)
    append_msg("user", user, header=current_user_header)
    append_msg("tool", tool)
    return records, diagnostics


def parse_aider_input_history(
    provider: str,
    project: str,
    file_path: Path,
    *,
    text: str | None = None,
) -> tuple[list[MemoryRecord], list[str]]:
    """Parse prompt_toolkit ``FileHistory`` input history as user-only turns."""
    diagnostics: list[str] = []
    if text is None:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            diagnostics.append(f"read_error:{type(exc).__name__}")
            return [], diagnostics

    conversation_id = conversation_id_for_path(file_path)
    records: list[MemoryRecord] = []
    entry_lines: list[str] = []
    entry_ts: str | None = None
    turn_ordinal = 0

    def flush_entry() -> None:
        nonlocal turn_ordinal, entry_lines, entry_ts
        if not entry_lines:
            entry_ts = None
            return
        content = "\n".join(entry_lines).strip("\n")
        entry_lines = []
        if not content.strip():
            entry_ts = None
            return
        turn_ordinal += 1
        if entry_ts:
            base_iso, origin = entry_ts, TIMESTAMP_ORIGIN_INPUT_COMMENT
        else:
            base_iso, origin = SYNTHETIC_EPOCH_ISO, TIMESTAMP_ORIGIN_SYNTHETIC
        ts, ts_meta = _stable_turn_timestamp(
            base_iso=base_iso, turn_ordinal=turn_ordinal, origin=origin
        )
        meta = {
            "source": "aider-input-history",
            "source_filename": file_path.name,
            "turn_ordinal": turn_ordinal,
            "limitations": "user_input_only",
            **ts_meta,
        }
        sk = source_key_for_turn(
            conversation_id=conversation_id,
            turn_ordinal=turn_ordinal,
            role="user",
            content=content,
        )
        records.append(
            MemoryRecord(
                provider=provider,
                project=project,
                conversation_id=conversation_id,
                role="user",
                content=content,
                timestamp=ts,
                metadata_json=json.dumps(meta, ensure_ascii=True),
                source_key=sk,
            )
        )
        entry_ts = None

    for line_no, raw in enumerate(text.splitlines(), start=1):
        try:
            if raw.startswith("#"):
                flush_entry()
                rest = raw[1:].strip()
                entry_ts = _parse_session_timestamp(rest) if rest else None
                continue
            if raw.startswith("+"):
                entry_lines.append(raw[1:])
                continue
            if raw.strip() == "":
                continue
            diagnostics.append(f"line_{line_no}:skipped_malformed")
        except Exception as exc:
            diagnostics.append(f"line_{line_no}:{type(exc).__name__}")
            continue
    flush_entry()
    return records, diagnostics


def parse_aider_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    """Route Aider files by evidenced filename / format markers."""
    name = file_path.name.lower()
    if name == ".aider.input.history" or name.endswith(".input.history"):
        records, diags = parse_aider_input_history(provider, project, file_path)
        for d in diags:
            logger.warning("aider input-history %s: %s", file_path.name, d)
        return records

    if (
        name == ".aider.chat.history.md"
        or name.endswith(".chat.history.md")
        or file_path.suffix.lower() in {".md", ".markdown"}
    ):
        records, diags = parse_aider_chat_history(provider, project, file_path)
        for d in diags:
            logger.warning("aider chat-history %s: %s", file_path.name, d)
        return records

    logger.warning("aider parser skipping unrecognized file type: %s", file_path.name)
    return []
