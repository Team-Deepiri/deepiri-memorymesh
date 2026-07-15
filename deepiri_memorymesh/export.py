from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from typing import Any, Literal

from .models import now_iso

ExportFormat = Literal["txt", "md", "markdown", "json"]


def normalize_format(fmt: str) -> ExportFormat:
    key = fmt.strip().lower()
    if key in ("md", "markdown"):
        return "md"
    if key in ("txt", "text", "plain"):
        return "txt"
    if key == "json":
        return "json"
    raise ValueError(f"Unsupported export format: {fmt!r} (use txt, md, or json)")


def gather_project_export(
    *,
    project: str,
    messages: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    agent_state: list[dict[str, Any]],
    stats: dict[str, int],
    provider: str | None = None,
) -> dict[str, Any]:
    filtered = messages
    if provider:
        p = provider.strip().lower()
        filtered = [m for m in messages if str(m.get("provider", "")).lower() == p]
        conv_ids = {str(m.get("conversation_id")) for m in filtered}
        summaries = [
            s for s in summaries if str(s.get("conversation_id")) in conv_ids
        ]
    return {
        "project": project,
        "exported_at": now_iso(),
        "provider_filter": provider,
        "stats": stats,
        "messages": filtered,
        "summaries": summaries,
        "agent_state": agent_state,
    }


def render_export(payload: dict[str, Any], fmt: ExportFormat) -> str:
    if fmt == "json":
        return json.dumps(payload, ensure_ascii=False, indent=2)
    if fmt == "md":
        return _render_markdown(payload)
    return _render_txt(payload)


def _render_markdown(payload: dict[str, Any]) -> str:
    project = payload["project"]
    lines = [
        f"# MemoryMesh Export: {project}",
        "",
        f"**Exported:** {payload['exported_at']}",
    ]
    if payload.get("provider_filter"):
        lines.append(f"**Provider filter:** {payload['provider_filter']}")
    stats = payload.get("stats") or {}
    lines.extend(
        [
            "",
            "## Stats",
            "",
            f"- Messages: {stats.get('messages', 0)}",
            f"- Conversations: {stats.get('conversations', 0)}",
            f"- Summaries: {stats.get('summaries', 0)}",
            f"- Embeddings: {stats.get('embeddings', 0)}",
            "",
        ]
    )
    lines.extend(_format_messages_md(payload.get("messages") or []))
    lines.extend(_format_summaries_md(payload.get("summaries") or []))
    lines.extend(_format_agent_state_md(payload.get("agent_state") or []))
    return "\n".join(lines).rstrip() + "\n"


def _render_txt(payload: dict[str, Any]) -> str:
    project = payload["project"]
    lines = [
        f"MemoryMesh Export: {project}",
        f"Exported: {payload['exported_at']}",
    ]
    if payload.get("provider_filter"):
        lines.append(f"Provider filter: {payload['provider_filter']}")
    stats = payload.get("stats") or {}
    lines.extend(
        [
            "",
            "=== Stats ===",
            f"Messages: {stats.get('messages', 0)}",
            f"Conversations: {stats.get('conversations', 0)}",
            f"Summaries: {stats.get('summaries', 0)}",
            f"Embeddings: {stats.get('embeddings', 0)}",
            "",
        ]
    )
    lines.extend(_format_messages_txt(payload.get("messages") or []))
    lines.extend(_format_summaries_txt(payload.get("summaries") or []))
    lines.extend(_format_agent_state_txt(payload.get("agent_state") or []))
    return "\n".join(lines).rstrip() + "\n"


def _group_messages(messages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for msg in messages:
        conv_id = str(msg.get("conversation_id") or "unknown")
        grouped[conv_id].append(msg)
    return grouped


def _format_messages_md(messages: list[dict[str, Any]]) -> list[str]:
    if not messages:
        return ["## Conversations", "", "_No messages._", ""]
    lines = ["## Conversations", ""]
    for conv_id, msgs in _group_messages(messages).items():
        provider = str(msgs[0].get("provider") or "unknown")
        lines.append(f"### {conv_id}")
        lines.append(f"_Provider: {provider}_")
        lines.append("")
        for msg in msgs:
            role = str(msg.get("role") or "unknown")
            ts = str(msg.get("timestamp") or "")
            content = str(msg.get("content") or "")
            lines.append(f"#### {role}" + (f" ({ts})" if ts else ""))
            lines.append("")
            lines.append(content)
            lines.append("")
    return lines


def _format_messages_txt(messages: list[dict[str, Any]]) -> list[str]:
    if not messages:
        return ["=== Conversations ===", "(no messages)", ""]
    lines = ["=== Conversations ===", ""]
    for conv_id, msgs in _group_messages(messages).items():
        provider = str(msgs[0].get("provider") or "unknown")
        lines.append(f"--- {conv_id} (provider: {provider}) ---")
        lines.append("")
        for msg in msgs:
            role = str(msg.get("role") or "unknown")
            ts = str(msg.get("timestamp") or "")
            header = f"[{role}]"
            if ts:
                header += f" {ts}"
            lines.append(header)
            lines.append(str(msg.get("content") or ""))
            lines.append("")
    return lines


def _format_summaries_md(summaries: list[dict[str, Any]]) -> list[str]:
    lines = ["## Summaries", ""]
    if not summaries:
        lines.append("_No summaries._")
        lines.append("")
        return lines
    for row in summaries:
        conv_id = str(row.get("conversation_id") or "unknown")
        method = str(row.get("method") or "")
        created = str(row.get("created_at") or "")
        lines.append(f"### {conv_id}")
        if method or created:
            lines.append(f"_Method: {method}_ | _Created: {created}_")
        lines.append("")
        lines.append(str(row.get("summary") or ""))
        lines.append("")
    return lines


def _format_summaries_txt(summaries: list[dict[str, Any]]) -> list[str]:
    lines = ["=== Summaries ===", ""]
    if not summaries:
        lines.append("(no summaries)")
        lines.append("")
        return lines
    for row in summaries:
        conv_id = str(row.get("conversation_id") or "unknown")
        lines.append(f"--- {conv_id} ---")
        lines.append(str(row.get("summary") or ""))
        lines.append("")
    return lines


def _format_agent_state_md(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Agent state", ""]
    if not rows:
        lines.append("_No agent state._")
        lines.append("")
        return lines
    for row in rows:
        agent = str(row.get("agent") or "unknown")
        key = str(row.get("state_key") or row.get("key") or "")
        updated = str(row.get("updated_at") or "")
        lines.append(f"### {agent} / `{key}`")
        if updated:
            lines.append(f"_Updated: {updated}_")
        lines.append("")
        lines.append(str(row.get("value") or ""))
        lines.append("")
    return lines


def _format_agent_state_txt(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["=== Agent state ===", ""]
    if not rows:
        lines.append("(no agent state)")
        lines.append("")
        return lines
    for row in rows:
        agent = str(row.get("agent") or "unknown")
        key = str(row.get("state_key") or row.get("key") or "")
        lines.append(f"--- {agent} / {key} ---")
        lines.append(str(row.get("value") or ""))
        lines.append("")
    return lines


def copy_to_clipboard(text: str) -> bool:
    """Copy text using the first available system clipboard utility."""
    data = text.encode("utf-8")
    candidates: list[list[str]] = [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["pbcopy"],
    ]
    for cmd in candidates:
        if not shutil.which(cmd[0]):
            continue
        try:
            subprocess.run(cmd, input=data, check=True)
            return True
        except (OSError, subprocess.CalledProcessError):
            continue
    return False
