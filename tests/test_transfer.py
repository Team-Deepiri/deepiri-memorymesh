from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.models import MemoryRecord
from deepiri_memorymesh.storage import MemoryStore
from deepiri_memorymesh.sync_service import MemoryMesh
from deepiri_memorymesh.transfer_delivery import deliver_transfer_bundle
from deepiri_memorymesh.transfer_formats import (
    load_transfer_bundle,
    messages_from_bundle,
    render_markdown,
    render_provider_json,
)


@pytest.fixture
def mesh(tmp_path: Path) -> MemoryMesh:
    settings = Settings(
        db_path=tmp_path / "memorymesh.db",
        embedding_backend="fallback",
    )
    service = MemoryMesh(settings)
    service.init()
    service.store.insert_messages(
        [
            MemoryRecord(
                provider="claude",
                project="demo",
                conversation_id="c1",
                role="user",
                content="We are building MemoryMesh transfer.",
                timestamp="2026-01-01T00:00:00+00:00",
            ),
            MemoryRecord(
                provider="claude",
                project="demo",
                conversation_id="c1",
                role="assistant",
                content="Transfer should render markdown and deliver to inbox.",
                timestamp="2026-01-01T00:00:01+00:00",
            ),
        ]
    )
    return service


def test_transfer_bundle_and_delivery(mesh: MemoryMesh, tmp_path: Path) -> None:
    bundle_path, count, delivery = mesh.transfer(
        project="demo",
        from_provider="claude",
        to_provider="cursor",
        push_via_bridge=True,
    )
    assert count == 2
    assert bundle_path.exists()
    assert delivery is not None
    assert delivery.context_md.exists()
    assert delivery.import_json.exists()
    assert delivery.message_count == 2

    payload = load_transfer_bundle(bundle_path)
    assert payload["from_provider"] == "claude"
    assert payload["to_provider"] == "cursor"
    assert len(messages_from_bundle(payload)) == 2

    md = render_markdown(payload)
    assert "MemoryMesh transfer context" in md
    assert "building MemoryMesh transfer" in md

    cursor_json = render_provider_json(payload, "cursor")
    assert cursor_json["conversationId"].startswith("transfer-")
    assert len(cursor_json["messages"]) == 2


def test_go_transfer_writes_inbox(mesh: MemoryMesh, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inbox_root = tmp_path / "inbox"
    monkeypatch.setattr(
        "deepiri_memorymesh.transfer_delivery.DEFAULT_INBOX_ROOT",
        inbox_root,
    )
    bundle_path, delivery = mesh.go_transfer(
        project="demo",
        from_provider="claude",
        to_provider="gemini",
        sync_source=False,
        compress_first=False,
        copy_clipboard=False,
    )
    assert bundle_path.exists()
    assert delivery.inbox_dir == inbox_root / "gemini"
    assert "Paste this block" in delivery.context_md.read_text(encoding="utf-8")


def test_deliver_ingests_under_target_provider(mesh: MemoryMesh, tmp_path: Path) -> None:
    bundle = {
        "project": "demo",
        "from_provider": "claude",
        "to_provider": "cursor",
        "conversation_id": "transfer-claude-to-cursor",
        "messages": [
            {"role": "user", "content": "hello", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"role": "assistant", "content": "world", "timestamp": "2026-01-01T00:00:01+00:00"},
        ],
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    delivery = deliver_transfer_bundle(
        bundle_path=bundle_path,
        target="cursor",
        mesh=mesh,
        inbox_root=tmp_path / "inbox",
    )
    assert delivery.ingested == 2
    rows = mesh.store.list_messages_by_provider("demo", "cursor")
    assert len(rows) == 2
