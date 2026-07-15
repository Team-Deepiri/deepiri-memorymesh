"""Extract Cursor IDE conversations from state.vscdb SQLite databases."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterator

from ..models import MemoryRecord, now_iso
from .base import normalize_content, records_from_messages, safe_str


def _decode_value(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _open_sqlite_readonly(db_path: Path) -> sqlite3.Connection:
    """Open SQLite DB; copy WAL sidecars for consistent reads when Cursor is closed."""
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    wal = db_path.with_suffix(db_path.suffix + "-wal")
    shm = db_path.with_suffix(db_path.suffix + "-shm")
    if wal.exists() or shm.exists():
        tmp = Path(tempfile.mkdtemp(prefix="memorymesh-cursor-"))
        dest = tmp / "state.vscdb"
        shutil.copy2(db_path, dest)
        if wal.exists():
            shutil.copy2(wal, dest.with_suffix(dest.suffix + "-wal"))
        if shm.exists():
            shutil.copy2(shm, dest.with_suffix(dest.suffix + "-shm"))
        db_path = dest
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _kv_rows(conn: sqlite3.Connection, table: str) -> Iterator[tuple[str, str]]:
    try:
        cur = conn.execute(f"SELECT key, value FROM {table}")
    except sqlite3.Error:
        return
    for row in cur:
        key = safe_str(row["key"])
        val = _decode_value(row["value"])
        if key and val:
            yield key, val


def _role_from_type(msg_type: Any) -> str:
    t = safe_str(msg_type)
    if t in {"1", "user"}:
        return "user"
    if t in {"2", "assistant", "ai"}:
        return "assistant"
    return "unknown"


def _messages_from_composer_data(composer_id: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    headers = data.get("fullConversationHeadersOnly") or []
    conv_map = data.get("conversationMap") or {}
    if isinstance(headers, list) and conv_map:
        for hdr in headers:
            if not isinstance(hdr, dict):
                continue
            bubble_id = safe_str(hdr.get("bubbleId"))
            bubble = conv_map.get(bubble_id) if isinstance(conv_map, dict) else None
            if not isinstance(bubble, dict):
                continue
            text = normalize_content(bubble.get("text") or bubble.get("content"))
            if not text:
                continue
            role = _role_from_type(bubble.get("type") or hdr.get("type"))
            ts_ms = bubble.get("createdAt") or data.get("createdAt")
            ts = now_iso()
            if ts_ms:
                try:
                    from datetime import datetime, timezone

                    ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).isoformat()
                except (TypeError, ValueError, OSError):
                    pass
            msgs.append(
                {
                    "role": role,
                    "content": text,
                    "timestamp": ts,
                    "metadata": {
                        "source": "cursor-composerData",
                        "composer_id": composer_id,
                        "bubble_id": bubble_id,
                    },
                }
            )
    return msgs


def _messages_from_bubbles(
    composer_id: str,
    bubbles: dict[str, dict[str, Any]],
    headers: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    order: list[str] = []
    if headers:
        for hdr in headers:
            if isinstance(hdr, dict) and hdr.get("bubbleId"):
                order.append(safe_str(hdr["bubbleId"]))
    if not order:
        order = list(bubbles.keys())
    for bubble_id in order:
        bubble = bubbles.get(bubble_id)
        if not isinstance(bubble, dict):
            continue
        text = normalize_content(bubble.get("text") or bubble.get("content"))
        if not text:
            continue
        role = _role_from_type(bubble.get("type"))
        ts_ms = bubble.get("createdAt")
        ts = now_iso()
        if ts_ms:
            try:
                from datetime import datetime, timezone

                ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).isoformat()
            except (TypeError, ValueError, OSError):
                pass
        msgs.append(
            {
                "role": role,
                "content": text,
                "timestamp": ts,
                "metadata": {
                    "source": "cursor-bubbleId",
                    "composer_id": composer_id,
                    "bubble_id": bubble_id,
                },
            }
        )
    return msgs


def parse_cursor_sqlite(
    provider: str,
    project: str,
    db_path: Path,
    workspace_hint: str = "",
) -> list[MemoryRecord]:
    """Parse one Cursor state.vscdb file into memory records."""
    records: list[MemoryRecord] = []
    try:
        conn = _open_sqlite_readonly(db_path)
    except (FileNotFoundError, sqlite3.Error):
        return records

    composer_bubbles: dict[str, dict[str, dict[str, Any]]] = {}
    composer_headers: dict[str, list[dict[str, Any]]] = {}
    composer_meta: dict[str, dict[str, Any]] = {}

    with conn:
        for table in ("cursorDiskKV", "ItemTable"):
            for key, val in _kv_rows(conn, table):
                if key.startswith("bubbleId:"):
                    parts = key.split(":", 2)
                    if len(parts) != 3:
                        continue
                    _, composer_id, bubble_id = parts
                    try:
                        bubble = json.loads(val)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(bubble, dict):
                        composer_bubbles.setdefault(composer_id, {})[bubble_id] = bubble
                elif key.startswith("composerData:"):
                    composer_id = key.split(":", 1)[1]
                    try:
                        data = json.loads(val)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        composer_meta[composer_id] = data
                        headers = data.get("fullConversationHeadersOnly")
                        if isinstance(headers, list):
                            composer_headers[composer_id] = headers
                elif key == "composer.composerData" and table == "ItemTable":
                    try:
                        data = json.loads(val)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        for comp in data.get("allComposers") or []:
                            if isinstance(comp, dict) and comp.get("composerId"):
                                cid = safe_str(comp["composerId"])
                                composer_meta.setdefault(cid, comp)

    all_composer_ids = set(composer_meta) | set(composer_bubbles)
    for composer_id in all_composer_ids:
        meta = composer_meta.get(composer_id) or {}
        headers = composer_headers.get(composer_id)
        bubbles = composer_bubbles.get(composer_id) or {}
        msgs = _messages_from_bubbles(composer_id, bubbles, headers)
        if not msgs:
            msgs = _messages_from_composer_data(composer_id, meta)
        if not msgs:
            continue
        conv_name = safe_str(meta.get("name"), composer_id[:12])
        conv_id = f"{workspace_hint}:{conv_name}:{composer_id[:8]}" if workspace_hint else f"{conv_name}:{composer_id[:8]}"
        records.extend(records_from_messages(provider, project, conv_id, msgs))

    conn.close()
    return records


def find_cursor_databases(cursor_user_dir: Path | None = None) -> list[Path]:
    from ..device_paths import _cursor_user_dir

    base = cursor_user_dir or _cursor_user_dir()
    if not base.exists():
        return []
    dbs: list[Path] = []
    global_db = base / "globalStorage" / "state.vscdb"
    if global_db.exists():
        dbs.append(global_db)
    ws = base / "workspaceStorage"
    if ws.exists():
        dbs.extend(sorted(ws.glob("*/state.vscdb")))
    return dbs
