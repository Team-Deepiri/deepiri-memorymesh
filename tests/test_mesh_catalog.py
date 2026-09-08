"""Tests for the chat_catalog migration, catalog builder, and mesh find/index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.migrations import CURRENT_SCHEMA_VERSION, current_schema_version
from deepiri_memorymesh.models import MemoryRecord
from deepiri_memorymesh.sync_service import MemoryMesh


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "test.db",
        embedding_backend="fallback",
        providers=[],
    )


def _seed_project(mesh: MemoryMesh, project: str, provider: str, messages: list[str]) -> None:
    mesh.init()
    for i, content in enumerate(messages):
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider=provider,
                    project=project,
                    conversation_id=f"{project}-{provider}-conv",
                    role="user" if i % 2 == 0 else "assistant",
                    content=content,
                )
            ]
        )


def _schema_version(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return current_schema_version(conn)
    finally:
        conn.close()


class TestChatCatalogMigration:
    def test_migrates_to_v7(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        mesh.init()
        assert _schema_version(mesh.settings.db_path) == CURRENT_SCHEMA_VERSION
        assert CURRENT_SCHEMA_VERSION >= 7

    def test_idempotent_on_reinit(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        mesh.init()
        mesh.init()
        assert _schema_version(mesh.settings.db_path) == CURRENT_SCHEMA_VERSION


class TestRebuildCatalog:
    def test_empty_db(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        mesh.init()
        assert mesh.index_catalog() == 0
        assert mesh.catalog_stats() == {"conversations": 0, "projects": 0}

    def test_indexes_conversations(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "proj-a", "claude", ["hello there", "how are you"])
        count = mesh.index_catalog()
        assert count == 1
        stats = mesh.catalog_stats()
        assert stats == {"conversations": 1, "projects": 1}

    def test_cross_project_indexing(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "alpha", "claude", ["msg one"])
        _seed_project(mesh, "beta", "cursor", ["msg two"])
        count = mesh.index_catalog()
        assert count == 2
        stats = mesh.catalog_stats()
        assert stats["conversations"] == 2
        assert stats["projects"] == 2

    def test_scoped_to_project(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "alpha", "claude", ["msg one"])
        _seed_project(mesh, "beta", "cursor", ["msg two"])
        count = mesh.index_catalog(project="alpha")
        assert count == 1
        rows = mesh.find_conversations()
        projects = {r["project"] for r in rows}
        assert projects == {"alpha"}

    def test_rebuild_updates_existing_row(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "proj", "claude", ["first message"])
        mesh.index_catalog()
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="claude",
                    project="proj",
                    conversation_id="proj-claude-conv",
                    role="user",
                    content="second message",
                )
            ]
        )
        mesh.index_catalog()
        rows = mesh.find_conversations(project="proj")
        assert len(rows) == 1
        assert rows[0]["message_count"] == 2

    def test_title_from_earliest_message(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "proj", "claude", ["the very first message here"])
        mesh.index_catalog()
        rows = mesh.find_conversations(project="proj")
        assert rows[0]["title"] == "the very first message here"


class TestFindConversations:
    def test_empty_catalog(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        mesh.init()
        assert mesh.find_conversations() == []

    def test_filters_by_project(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "alpha", "claude", ["hi"])
        _seed_project(mesh, "beta", "cursor", ["hi"])
        mesh.index_catalog()
        rows = mesh.find_conversations(project="alpha")
        assert len(rows) == 1
        assert rows[0]["project"] == "alpha"

    def test_filters_by_provider(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "proj", "claude", ["hi"])
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="cursor",
                    project="proj",
                    conversation_id="proj-cursor-conv",
                    role="user",
                    content="hello",
                )
            ]
        )
        mesh.index_catalog()
        rows = mesh.find_conversations(provider="cursor")
        assert len(rows) == 1
        assert rows[0]["provider"] == "cursor"

    def test_filters_by_title_query(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        _seed_project(mesh, "proj-a", "claude", ["quantum computing topic"])
        _seed_project(mesh, "proj-b", "cursor", ["cooking recipes"])
        mesh.index_catalog()
        rows = mesh.find_conversations(query="quantum")
        assert len(rows) == 1
        assert rows[0]["project"] == "proj-a"

    def test_respects_limit(self, tmp_path: Path) -> None:
        mesh = MemoryMesh(_make_settings(tmp_path))
        for i in range(5):
            _seed_project(mesh, f"proj-{i}", "claude", ["hi"])
        mesh.index_catalog()
        rows = mesh.find_conversations(limit=2)
        assert len(rows) == 2
