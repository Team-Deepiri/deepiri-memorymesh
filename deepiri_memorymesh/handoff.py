"""Provider-agnostic conversation handoff.

Writes paste-ready handoff markdown into locations a target agent can read.
Every target resolves its destinations *dynamically* from a small registry
so nothing is hardcoded to a single provider — unknown providers fall back
to a generic per-provider dot-directory in the workspace.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .session_bridge import build_resume_pack, render_resume_brief
from .transfer_formats import load_transfer_bundle, render_markdown

HANDOFF_FILENAME = "memorymesh-handoff.md"
HANDOFF_META_FILENAME = "memorymesh-handoff.json"


from .workspace import workspace_slug
def _cursor_targets(workspace: Path) -> list[Path]:
    # Per-workspace dotdir + the machine-wide Cursor agent project dir.
    return [
        workspace / ".cursor" / HANDOFF_FILENAME,
        Path.home() / ".cursor" / "projects" / workspace_slug(workspace).lstrip("-") / HANDOFF_FILENAME,
    ]


def _claude_targets(workspace: Path) -> list[Path]:
    config = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(config).expanduser() if config else Path.home() / ".claude"
    return [
        workspace / ".claude" / HANDOFF_FILENAME,
        base / "projects" / workspace_slug(workspace) / HANDOFF_FILENAME,
    ]


def _opencode_targets(workspace: Path) -> list[Path]:
    return [
        workspace / ".opencode" / HANDOFF_FILENAME,
        Path.home() / ".config" / "opencode" / HANDOFF_FILENAME,
    ]


def _gemini_targets(workspace: Path) -> list[Path]:
    return [
        workspace / ".gemini" / HANDOFF_FILENAME,
        Path.home() / ".gemini" / HANDOFF_FILENAME,
    ]


def _continue_targets(workspace: Path) -> list[Path]:
    return [
        workspace / ".continue" / HANDOFF_FILENAME,
        Path.home() / ".continue" / HANDOFF_FILENAME,
    ]


def _aider_targets(workspace: Path) -> list[Path]:
    return [workspace / ".aider" / HANDOFF_FILENAME]


@dataclass(slots=True)
class HandoffTarget:
    key: str
    resolver: Callable[[Path], list[Path]]


# Registry of known providers. Anything not here uses the generic fallback,
# so the handoff mechanism works for any current or future provider.
HANDOFF_TARGETS: dict[str, HandoffTarget] = {
    "cursor": HandoffTarget("cursor", _cursor_targets),
    "claude": HandoffTarget("claude", _claude_targets),
    "anthropic": HandoffTarget("claude", _claude_targets),
    "opencode": HandoffTarget("opencode", _opencode_targets),
    "gemini": HandoffTarget("gemini", _gemini_targets),
    "continue": HandoffTarget("continue", _continue_targets),
    "aider": HandoffTarget("aider", _aider_targets),
}


def _generic_targets(provider: str, workspace: Path) -> list[Path]:
    """Fallback for unknown providers: a per-provider dotdir in the workspace."""
    safe = provider.strip().lower().replace("/", "-") or "provider"
    return [
        workspace / f".{safe}" / HANDOFF_FILENAME,
        Path.home() / ".config" / "deepiri-memorymesh" / "handoff" / safe / HANDOFF_FILENAME,
    ]


def resolve_handoff_paths(provider: str, workspace: Path) -> list[Path]:
    """Resolve destination handoff files for a provider, dynamically.

    Known providers use their registered resolver; unknown providers get a
    generic per-provider location. Never hardcoded to a single provider.
    """
    key = provider.strip().lower()
    workspace = workspace.expanduser().resolve()
    target = HANDOFF_TARGETS.get(key)
    if target is not None:
        return target.resolver(workspace)
    return _generic_targets(key, workspace)


def write_handoff_files(
    bundle_path: Path,
    workspace: Path,
    *,
    provider: str | None = None,
    max_chars: int = 24_000,
    use_resume_brief: bool = True,
) -> dict[str, Any]:
    """Write paste-ready handoff markdown for the bundle's target provider.

    The target provider is taken from ``provider`` when given, else from the
    bundle's ``to_provider`` field — so the destination is chosen dynamically
    per transfer rather than being fixed to one tool.

    By default writes a compressed *resume brief* (summary + tail turns) instead
    of dumping the full transcript — this is what makes cross-agent handoff usable.
    """
    bundle = load_transfer_bundle(bundle_path)
    target_provider = (provider or str(bundle.get("to_provider") or "")).strip().lower()
    if not target_provider:
        target_provider = "generic"
    if use_resume_brief:
        pack = build_resume_pack(bundle)
        content = render_resume_brief(pack, max_chars=max_chars)
    else:
        content = render_markdown(bundle, max_chars=max_chars)

    workspace = workspace.expanduser().resolve()
    dests = resolve_handoff_paths(target_provider, workspace)
    written: list[str] = []
    for dest in dests:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        written.append(str(dest))

    meta = {
        "provider": target_provider,
        "handoff_files": written,
        "workspace": str(workspace),
        "conversation_id": bundle.get("conversation_id"),
        "message_count": len(bundle.get("messages") or []),
        "project": bundle.get("project"),
        "from_provider": bundle.get("from_provider"),
        "to_provider": bundle.get("to_provider"),
    }
    if written:
        meta_path = Path(written[0]).parent / HANDOFF_META_FILENAME
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        meta["meta_file"] = str(meta_path)
    return meta
