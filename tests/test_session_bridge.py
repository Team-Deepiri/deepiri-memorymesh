from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepiri_memorymesh.handoff import resolve_handoff_paths, write_handoff_files
from deepiri_memorymesh.session_bridge import build_resume_pack, render_resume_brief
from deepiri_memorymesh.workspace import workspace_slug


def test_workspace_slug_normalization() -> None:
    slug = workspace_slug(Path("/home/user/my_project"))
    assert slug.startswith("-")
    assert "my-project" in slug or "my_project" in slug.replace("_", "-")


def test_resolve_handoff_paths_dynamic() -> None:
    ws = Path("/tmp/ws")
    cursor_paths = resolve_handoff_paths("cursor", ws)
    claude_paths = resolve_handoff_paths("claude", ws)
    unknown_paths = resolve_handoff_paths("some-new-agent", ws)
    assert any(".cursor" in str(p) for p in cursor_paths)
    assert any(".claude" in str(p) for p in claude_paths)
    assert any(".some-new-agent" in str(p) for p in unknown_paths)


def test_resume_pack_uses_tail_not_full_dump() -> None:
    messages = [
        {"role": "user", "content": f"message {i}", "timestamp": f"t{i}"}
        for i in range(30)
    ]
    bundle = {
        "project": "demo",
        "from_provider": "claude",
        "to_provider": "gemini",
        "conversation_id": "sess-1",
        "messages": messages,
        "summaries": [],
    }
    pack = build_resume_pack(bundle, tail_turns=5)
    assert pack["message_count"] == 30
    assert len(pack["tail"]) == 5
    assert pack["tail"][0]["content"] == "message 25"
    brief = render_resume_brief(pack)
    assert "message 29" in brief
    assert "Resume directive" in brief


def test_write_handoff_files_for_any_provider(tmp_path: Path) -> None:
    bundle = {
        "project": "demo",
        "from_provider": "claude",
        "to_provider": "gemini",
        "conversation_id": "c1",
        "messages": [
            {"role": "user", "content": "build coverage sim", "timestamp": "t1"},
            {"role": "assistant", "content": "opened PR #65", "timestamp": "t2"},
        ],
        "summaries": [],
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    ws = tmp_path / "workspace"
    ws.mkdir()
    meta = write_handoff_files(bundle_path, ws, provider="gemini")
    assert meta["provider"] == "gemini"
    assert meta["handoff_files"]
    text = Path(meta["handoff_files"][0]).read_text(encoding="utf-8")
    assert "PR #65" in text
    assert "session resume" in text.lower()
