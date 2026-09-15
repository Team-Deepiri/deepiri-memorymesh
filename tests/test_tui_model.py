"""Tests for the TUI session model (pure data layer over the SDK)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.models import MemoryRecord
from deepiri_memorymesh.scanner import DeviceSession
from deepiri_memorymesh.sync_service import MemoryMesh
from deepiri_memorymesh.transfer_delivery import DEFAULT_INBOX_ROOT
from deepiri_memorymesh.tui.session_model import (
    SessionRow,
    build_session_list,
    conversation_messages,
    export_conversation,
    filter_sessions,
    transfer_destinations,
    transfer_session,
)

CLAUDE_LINE = (
    json.dumps(
        {
            "type": "user",
            "sessionId": "dsk1",
            "timestamp": "2026-01-02T00:00:00+00:00",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "hello from a disk file"}],
            },
        }
    )
    + "\n"
)


def _mesh(tmp_path: Path) -> MemoryMesh:
    settings = Settings(db_path=tmp_path / "m.db", embedding_backend="fallback")
    service = MemoryMesh(settings)
    service.init()
    service.store.insert_messages(
        [
            MemoryRecord(
                provider="claude",
                project="demo",
                conversation_id="c1",
                role="user",
                content="How does transfer work?",
                timestamp="2026-01-01T00:00:00+00:00",
            ),
            MemoryRecord(
                provider="claude",
                project="demo",
                conversation_id="c1",
                role="assistant",
                content="It renders a bundle and delivers it to the inbox.",
                timestamp="2026-01-01T00:00:01+00:00",
            ),
            MemoryRecord(
                provider="opencode",
                project="demo",
                conversation_id="c2",
                role="user",
                content="What is memorymesh status?",
                timestamp="2026-01-03T00:00:00+00:00",
            ),
        ]
    )
    return service


def _row(provider: str, conversation_id: str, source: str = "db", **kw) -> SessionRow:
    return SessionRow(
        provider=provider,
        conversation_id=conversation_id,
        source=source,
        **kw,
    )


def test_build_session_list_merges_db_and_disk(tmp_path: Path) -> None:
    mesh = _mesh(tmp_path)
    disk = [
        DeviceSession(
            provider="claude",
            conversation_id="c1",
            path=tmp_path / "c1.jsonl",
            workspace="demo",
            mtime=1.0,
            preview="on disk clone",
        ),
        DeviceSession(
            provider="opencode",
            conversation_id="c3",
            path=tmp_path / "novel.jsonl",
            workspace="elsewhere",
            mtime=2.0,
            preview="never ingested",
        ),
    ]
    rows = build_session_list(mesh, disk_sessions=disk)

    by_key = {(r.provider, r.conversation_id): r for r in rows}
    assert len(rows) == 3
    merged = by_key[("claude", "c1")]
    assert merged.source == "db+disk"
    assert merged.path == tmp_path / "c1.jsonl"
    assert merged.message_count == 2
    assert merged.project == "demo"
    assert "How does transfer work?" in merged.last_user_preview

    disk_only = by_key[("opencode", "c3")]
    assert disk_only.source == "disk"
    assert disk_only.project == ""

    assert rows[0].conversation_id == "c2"  # newest mtime first


def test_merge_prefers_richer_db_row(tmp_path: Path) -> None:
    mesh = _mesh(tmp_path)
    disk = [
        DeviceSession(
            provider="claude",
            conversation_id="c1",
            path=tmp_path / "c1.jsonl",
            workspace="demo",
            mtime=99.0,
            preview="stale disk",
        )
    ]
    rows = build_session_list(mesh, disk_sessions=disk)
    row = next(r for r in rows if r.conversation_id == "c1")
    assert row.source == "db+disk"
    assert row.message_count == 2
    assert row.last_user_preview == "How does transfer work?"


def test_filter_sessions(tmp_path: Path) -> None:
    mesh = _mesh(tmp_path)
    rows = build_session_list(mesh, disk_sessions=[])
    assert len(filter_sessions(rows, provider="opencode")) == 1
    assert {r.provider for r in filter_sessions(rows, text="memorymesh status")} == {
        "opencode"
    }
    assert len(filter_sessions(rows, provider="claude", text="transfer")) == 1
    assert len(filter_sessions(rows, text="no such text")) == 0


def test_conversation_messages_reads_db(tmp_path: Path) -> None:
    mesh = _mesh(tmp_path)
    rows = build_session_list(mesh)
    row = next(r for r in rows if r.conversation_id == "c1")
    messages = conversation_messages(mesh, row)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"


def test_conversation_messages_cursor_plain_text(tmp_path: Path) -> None:
    mesh = _mesh(tmp_path)
    transcript = tmp_path / "transcripts" / "cts.txt"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(
        "USER: plain text assistant question\n"
        "ASSISTANT: plain text assistant answer\n",
        encoding="utf-8",
    )
    disk = DeviceSession(
        provider="cursor",
        conversation_id="cts",
        path=transcript,
        workspace="demo",
        mtime=5.0,
        preview="plain text assistant question",
    )
    rows = build_session_list(mesh, disk_sessions=[disk])
    row = next(r for r in rows if r.conversation_id == "cts")
    messages = conversation_messages(mesh, row)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "plain text assistant question"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "plain text assistant answer"


def test_transfer_db_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.helpers import temp_home

    mesh = _mesh(tmp_path)
    inbox = tmp_path / "inbox"
    monkeypatch.setattr("deepiri_memorymesh.transfer_delivery.DEFAULT_INBOX_ROOT", inbox)
    with temp_home(tmp_path / "home"):
        rows = build_session_list(mesh)
        row = next(r for r in rows if r.conversation_id == "c1")
        outcome = transfer_session(mesh, row, "opencode", project="demo")

    assert outcome.message_count == 2
    assert outcome.bundle_path.exists()
    assert outcome.ingested_from_disk is False
    assert (inbox / "opencode" / "context.md").exists()
    assert (inbox / "opencode" / "import.json").exists()


def test_transfer_disk_only_ingests_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.helpers import temp_home

    mesh = _mesh(tmp_path)
    transcript = tmp_path / "transcripts" / "dsk1.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(CLAUDE_LINE, encoding="utf-8")
    inbox = tmp_path / "inbox"
    monkeypatch.setattr("deepiri_memorymesh.transfer_delivery.DEFAULT_INBOX_ROOT", inbox)
    disk = DeviceSession(
        provider="claude",
        conversation_id="dsk1",
        path=transcript,
        workspace="nowhere",
        mtime=5.0,
        preview="hello from a disk file",
    )
    with temp_home(tmp_path / "home"):
        rows = build_session_list(mesh, disk_sessions=[disk])
        row = next(r for r in rows if r.conversation_id == "dsk1")
        assert row.source == "disk"
        outcome = transfer_session(mesh, row, "opencode", project="demo")

    assert outcome.ingested_from_disk is True
    assert outcome.message_count == 1
    stored = mesh.store.list_messages_for_namespace(
        project="demo", provider="claude", conversation_id="dsk1"
    )
    assert len(stored) == 1
    assert stored[0]["content"] == "hello from a disk file"


def test_transfer_destinations_exclude_source(tmp_path: Path) -> None:
    mesh = _mesh(tmp_path)
    targets = transfer_destinations(mesh, "claude")
    assert "claude" not in targets
    assert "opencode" in targets
    assert "cursor" in targets


def test_export_conversation(tmp_path: Path) -> None:
    mesh = _mesh(tmp_path)
    rows = build_session_list(mesh)
    row = next(r for r in rows if r.conversation_id == "c2")
    md, slug = export_conversation(mesh, row, fmt="md")
    assert "opencode session" in md
    assert "memorymesh status" in md
    assert slug == "opencode-c2"
    as_json, _ = export_conversation(mesh, row, fmt="json")
    payload = json.loads(as_json)
    assert payload["conversation_id"] == "c2"
    assert payload["messages"][0]["role"] == "user"


def test_build_session_list_empty_db_returns_disk_rows(tmp_path: Path) -> None:
    mesh = MemoryMesh(Settings(db_path=tmp_path / "empty.db", embedding_backend="fallback"))
    disk = [
        DeviceSession(
            provider="gemini",
            conversation_id="g1",
            path=tmp_path / "g1.json",
            workspace="side",
            mtime=3.0,
            preview="standalone",
        )
    ]
    rows = build_session_list(mesh, disk_sessions=disk)
    assert len(rows) == 1
    assert rows[0].source == "disk"

    assert DEFAULT_INBOX_ROOT is not None  # import sanity for patch targets