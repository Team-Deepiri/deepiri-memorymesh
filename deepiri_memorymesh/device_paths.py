"""Platform-aware discovery of AI coding assistant data directories."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProviderRoot:
    provider: str
    path: Path
    kind: str  # json_tree | sqlite | mixed
    description: str
    globs: list[str] = field(default_factory=list)
    exists: bool = False
    file_count: int = 0


def _home() -> Path:
    return Path.home()


def _cursor_user_dir() -> Path:
    system = platform.system()
    if system == "Darwin":
        return _home() / "Library/Application Support/Cursor/User"
    if system == "Windows":
        appdata = os.environ.get("APPDATA", str(_home() / "AppData/Roaming"))
        return Path(appdata) / "Cursor/User"
    return _home() / ".config/Cursor/User"


def _claude_roots() -> list[Path]:
    config = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(config).expanduser() if config else _home() / ".claude"
    return [
        base,
        base / "projects",
        base / "history.jsonl",
    ]


def _opencode_roots() -> list[Path]:
    data = os.environ.get("OPENCODE_DATA_DIR")
    base = Path(data).expanduser() if data else _home() / ".local/share/opencode"
    config = _home() / ".config/opencode"
    return [base, base / "project", config]


def discover_provider_roots() -> list[ProviderRoot]:
    """Return known on-disk locations for Claude Code, Cursor, and OpenCode."""
    roots: list[ProviderRoot] = []

    claude_base = _claude_roots()[0]
    claude_projects = claude_base / "projects"
    claude_history = claude_base / "history.jsonl"
    roots.append(
        ProviderRoot(
            provider="claude",
            path=claude_projects,
            kind="json_tree",
            description="Claude Code session transcripts (JSONL per project hash)",
            globs=["**/*.jsonl", "**/*.json"],
        )
    )
    if claude_history.exists():
        roots.append(
            ProviderRoot(
                provider="claude",
                path=claude_history,
                kind="jsonl",
                description="Claude Code global command history index",
                globs=[],
            )
        )

    cursor_user = _cursor_user_dir()
    roots.append(
        ProviderRoot(
            provider="cursor",
            path=cursor_user / "globalStorage" / "state.vscdb",
            kind="sqlite",
            description="Cursor global chat DB (bubbleId, composerData)",
            globs=[],
        )
    )
    roots.append(
        ProviderRoot(
            provider="cursor",
            path=cursor_user / "workspaceStorage",
            kind="sqlite_tree",
            description="Cursor per-workspace state.vscdb files",
            globs=["**/state.vscdb"],
        )
    )
    agent_transcripts = _home() / ".cursor/projects"
    roots.append(
        ProviderRoot(
            provider="cursor",
            path=agent_transcripts,
            kind="json_tree",
            description="Cursor agent plain-text transcripts",
            globs=["**/*.txt", "**/*.json", "**/*.jsonl"],
        )
    )

    for oc_path in _opencode_roots():
        roots.append(
            ProviderRoot(
                provider="opencode",
                path=oc_path,
                kind="mixed",
                description="OpenCode session storage (CLI + project dirs)",
                globs=["**/*.json", "**/*.jsonl", "**/storage/**"],
            )
        )

    for root in roots:
        p = root.path.expanduser()
        root.path = p
        root.exists = p.exists()
        if root.exists and p.is_file():
            root.file_count = 1
        elif root.exists and p.is_dir():
            if root.kind == "sqlite":
                root.file_count = 1 if p.is_file() else 0
            elif root.globs:
                count = 0
                for g in root.globs:
                    count += len(list(p.rglob(g.lstrip("/"))))
                root.file_count = count
            else:
                root.file_count = sum(1 for _ in p.rglob("*") if _.is_file())

    return roots


def primary_paths_dict() -> dict[str, str]:
    """Default provider_paths for config.yaml from discovered roots."""
    mapping: dict[str, str] = {}
    for root in discover_provider_roots():
        key = root.provider
        if key in mapping:
            continue
        if root.exists:
            mapping[key] = str(root.path if root.kind != "sqlite" else root.path.parent.parent)
    if "claude" not in mapping:
        mapping["claude"] = str(_claude_roots()[0])
    if "cursor" not in mapping:
        mapping["cursor"] = str(_cursor_user_dir())
    if "opencode" not in mapping:
        mapping["opencode"] = str(_opencode_roots()[0])
    return mapping
