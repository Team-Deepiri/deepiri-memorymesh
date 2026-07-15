from __future__ import annotations

from deepiri_memorymesh.export import gather_project_export, normalize_format, render_export


def test_normalize_format_aliases() -> None:
    assert normalize_format("markdown") == "md"
    assert normalize_format("text") == "txt"
    assert normalize_format("json") == "json"


def test_render_markdown_includes_messages_and_state() -> None:
    payload = gather_project_export(
        project="demo",
        messages=[
            {
                "provider": "cursor",
                "conversation_id": "c1",
                "role": "user",
                "content": "hello",
                "timestamp": "2020-01-01T00:00:00+00:00",
            }
        ],
        summaries=[{"conversation_id": "c1", "summary": "greeting", "method": "x", "created_at": "t"}],
        agent_state=[{"agent": "bot", "state_key": "k", "value": "v", "updated_at": "t"}],
        stats={"messages": 1, "conversations": 1, "summaries": 1, "embeddings": 0},
    )
    md = render_export(payload, "md")
    assert "# MemoryMesh Export: demo" in md
    assert "hello" in md
    assert "greeting" in md
    assert "bot" in md


def test_provider_filter_limits_messages() -> None:
    payload = gather_project_export(
        project="demo",
        messages=[
            {"provider": "cursor", "conversation_id": "c1", "role": "user", "content": "a"},
            {"provider": "claude", "conversation_id": "c2", "role": "user", "content": "b"},
        ],
        summaries=[],
        agent_state=[],
        stats={"messages": 2, "conversations": 2, "summaries": 0, "embeddings": 0},
        provider="cursor",
    )
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["content"] == "a"
