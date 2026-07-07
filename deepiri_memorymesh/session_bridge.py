"""Workspace Session Bridge — correlate provider sessions to a workspace path.

Novel approach: instead of dumping an entire project memory layer (10k+ msgs)
or requiring a manual conversation id, we:

1. Normalize the workspace to a slug shared by Claude/Cursor/OpenCode project dirs
2. Discover on-disk session files for that slug (mtime-ordered)
3. Pick the best session (latest activity in that workspace)
4. Build a *Resume Pack* — compressed summary + tail turns + agent resume directive
5. Deliver via the provider handoff registry (dynamic, not hardcoded)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compression import compress_conversation
from .workspace import workspace_slug


@dataclass(slots=True)
class SessionMatch:
    provider: str
    conversation_id: str
    source_path: Path | None
    mtime: float
    preview: str
    score: float


def _provider_project_dirs(provider: str, workspace: Path) -> list[Path]:
    """Return candidate on-disk project directories for a provider + workspace."""
    slug = workspace_slug(workspace)
    home = Path.home()
    key = provider.strip().lower()

    if key in {"claude", "anthropic"}:
        config = os.environ.get("CLAUDE_CONFIG_DIR")
        base = Path(config).expanduser() if config else home / ".claude"
        return [base / "projects" / slug]

    if key == "cursor":
        # Cursor uses slug without leading dash in .cursor/projects
        cursor_slug = slug.lstrip("-")
        return [home / ".cursor" / "projects" / cursor_slug]

    if key == "opencode":
        data = os.environ.get("OPENCODE_DATA_DIR")
        base = Path(data).expanduser() if data else home / ".local/share/opencode"
        return [base / "project" / slug.lstrip("-"), home / ".config/opencode"]

    # Generic: allow any provider to opt in via env or config path convention
    env_key = f"{key.upper()}_PROJECTS_DIR"
    if env_key in os.environ:
        return [Path(os.environ[env_key]).expanduser() / slug.lstrip("-")]
    return [home / ".config" / "deepiri-memorymesh" / "sessions" / key / slug.lstrip("-")]


def discover_sessions_on_disk(provider: str, workspace: Path) -> list[SessionMatch]:
    """Scan provider project dirs for session transcript files tied to workspace."""
    matches: list[SessionMatch] = []
    for root in _provider_project_dirs(provider, workspace):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".jsonl", ".json", ".txt"}:
                continue
            if path.name.startswith("."):
                continue
            preview = ""
            try:
                if path.suffix.lower() == ".jsonl":
                    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:8]:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        if isinstance(item, dict):
                            preview = _preview_from_record(item)
                            if preview:
                                break
                else:
                    preview = path.read_text(encoding="utf-8", errors="replace")[:200]
            except (json.JSONDecodeError, OSError):
                preview = path.name
            mtime = path.stat().st_mtime
            conv_id = path.stem
            if path.parent.name == "agent-transcripts" and path.parent.parent.name:
                # Cursor agent transcript folder layout
                conv_id = path.parent.name if path.suffix == ".jsonl" else path.stem
            matches.append(
                SessionMatch(
                    provider=provider.strip().lower(),
                    conversation_id=conv_id,
                    source_path=path,
                    mtime=mtime,
                    preview=preview.replace("\n", " ")[:160],
                    score=mtime,
                )
            )
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def _preview_from_record(item: dict[str, Any]) -> str:
    for key in ("display", "content", "text"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    msg = item.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
            return " ".join(parts).strip()
    return ""


def pick_best_session(
    *,
    workspace: Path,
    provider: str,
    store=None,
    project: str | None = None,
) -> SessionMatch | None:
    """Pick the best source session for a workspace.

    Priority: on-disk session files for workspace slug (newest), then DB
    conversations whose id or metadata matches the workspace slug.
    """
    on_disk = discover_sessions_on_disk(provider, workspace)
    if on_disk:
        return on_disk[0]

    if store is None or not project:
        return None

    slug = workspace_slug(workspace)
    slug_tail = slug.lstrip("-")
    rows = store.list_conversations(project=project, provider=provider.strip().lower(), limit=50)
    best: SessionMatch | None = None
    for row in rows:
        conv = str(row["conversation_id"])
        preview = str(row["last_user_preview"] or "")
        hay = f"{conv} {preview}".lower()
        if slug_tail.lower() not in hay and slug.lower() not in hay:
            # Still allow if this is simply the newest non-transfer conversation
            if conv.startswith("transfer-"):
                continue
        try:
            ts = str(row["last_timestamp"] or "")
            score = float(row["message_count"] or 0)
        except (TypeError, ValueError):
            score = 0.0
        candidate = SessionMatch(
            provider=provider.strip().lower(),
            conversation_id=conv,
            source_path=None,
            mtime=score,
            preview=preview[:160],
            score=score,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def build_resume_pack(
    bundle: dict[str, Any],
    *,
    tail_turns: int = 12,
    summary_chars: int = 2000,
) -> dict[str, Any]:
    """Build structured resume pack from a transfer bundle."""
    messages = bundle.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    normalized: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        normalized.append(
            {
                "role": str(msg.get("role") or "unknown"),
                "content": content,
                "timestamp": str(msg.get("timestamp") or ""),
            }
        )

    full_text = "\n".join(f'{m["role"]}: {m["content"]}' for m in normalized)
    summary = ""
    summaries = bundle.get("summaries") or []
    if isinstance(summaries, list):
        for item in summaries:
            if isinstance(item, dict) and str(item.get("summary") or "").strip():
                summary = str(item["summary"]).strip()
                break
    if not summary and full_text:
        summary = compress_conversation(full_text, target_chars=summary_chars)

    tail = normalized[-tail_turns:] if tail_turns > 0 else normalized

    return {
        "project": bundle.get("project"),
        "from_provider": bundle.get("from_provider"),
        "to_provider": bundle.get("to_provider"),
        "conversation_id": bundle.get("conversation_id"),
        "message_count": len(normalized),
        "summary": summary,
        "tail": tail,
        "resume_directive": (
            "Continue this work in the target agent. Treat the summary as long-term "
            "memory and the tail as the immediate thread. Do not restart from scratch."
        ),
    }


def render_resume_brief(pack: dict[str, Any], *, max_chars: int = 24_000) -> str:
    """Render agent-ready resume brief (not a full transcript dump)."""
    lines = [
        "# MemoryMesh session resume\n",
        f"- project: `{pack.get('project')}`\n",
        f"- from: `{pack.get('from_provider')}` → to: `{pack.get('to_provider')}`\n",
        f"- conversation: `{pack.get('conversation_id')}`\n",
        f"- messages in session: {pack.get('message_count')}\n\n",
        "## Resume directive\n\n",
        f"{pack.get('resume_directive')}\n\n",
    ]
    summary = str(pack.get("summary") or "").strip()
    if summary:
        lines.extend(["## Compressed summary\n\n", summary, "\n\n"])
    lines.append("## Recent turns (tail)\n\n")
    for msg in pack.get("tail") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "unknown").upper()
        block = f"### {role}\n\n{msg.get('content', '')}\n\n"
        if sum(len(x) for x in lines) + len(block) > max_chars:
            lines.append("\n_(tail truncated — full bundle in inbox import.json)_\n")
            break
        lines.append(block)
    return "".join(lines)
