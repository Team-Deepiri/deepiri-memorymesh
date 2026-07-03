from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROLE_ALIASES = {
    "human": "user",
    "ai": "assistant",
    "model": "assistant",
    "bot": "assistant",
    "system": "system",
    "tool": "tool",
}


def normalize_role(role: str) -> str:
    key = (role or "unknown").strip().lower()
    return ROLE_ALIASES.get(key, key)


def load_transfer_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("transfer bundle must be a JSON object")
    if "messages" not in payload:
        raise ValueError("transfer bundle missing messages")
    return payload


def messages_from_bundle(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    raw = bundle.get("messages") or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        role = normalize_role(str(item.get("role") or "unknown"))
        out.append(
            {
                "role": role,
                "content": content,
                "timestamp": str(item.get("timestamp") or ""),
                "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            }
        )
    return out


def _header(bundle: dict[str, Any]) -> str:
    project = str(bundle.get("project") or "default")
    source = str(bundle.get("from_provider") or "unknown")
    target = str(bundle.get("to_provider") or "unknown")
    conv = str(bundle.get("conversation_id") or "transfer")
    return (
        f"# MemoryMesh transfer context\n\n"
        f"- project: `{project}`\n"
        f"- from: `{source}` → to: `{target}`\n"
        f"- conversation: `{conv}`\n\n"
        f"Paste this block into your {target} chat as the first message "
        f"or system context so the session continues with prior context.\n\n"
        f"---\n\n"
    )


def render_markdown(bundle: dict[str, Any], max_chars: int = 60000) -> str:
    messages = messages_from_bundle(bundle)
    lines = [_header(bundle)]
    summaries = bundle.get("summaries") or []
    if isinstance(summaries, list) and summaries:
        lines.append("## Compressed summaries\n\n")
        for item in summaries:
            if not isinstance(item, dict):
                continue
            conv = str(item.get("conversation_id") or "conversation")
            summary = str(item.get("summary") or "").strip()
            if not summary:
                continue
            lines.append(f"### {conv}\n\n{summary}\n\n")
        lines.append("---\n\n")
    lines.append("## Conversation\n")
    for msg in messages:
        role = msg["role"].upper()
        block = f"### {role}\n\n{msg['content']}\n\n"
        if sum(len(x) for x in lines) + len(block) > max_chars:
            lines.append("\n_(truncated — see import.json for full transfer)_\n")
            break
        lines.append(block)
    return "".join(lines)


def render_provider_json(bundle: dict[str, Any], target: str) -> dict[str, Any]:
    key = target.strip().lower()
    messages = messages_from_bundle(bundle)
    conv_id = str(bundle.get("conversation_id") or f"memorymesh-transfer-{key}")
    normalized = [
        {
            "role": msg["role"],
            "content": msg["content"],
            **({"timestamp": msg["timestamp"]} if msg["timestamp"] else {}),
        }
        for msg in messages
    ]

    if key == "claude":
        return {
            "conversation_id": conv_id,
            "messages": normalized,
            "source": "memorymesh-transfer",
        }

    if key == "cursor":
        return {
            "conversationId": conv_id,
            "messages": [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "type": msg["role"],
                }
                for msg in messages
            ],
            "source": "memorymesh-transfer",
        }

    if key == "gemini":
        return {
            "session_id": conv_id,
            "messages": normalized,
            "source": "memorymesh-transfer",
        }

    if key in {"continue", "opencode", "aider"}:
        return {
            "conversation_id": conv_id,
            "messages": normalized,
            "provider": key,
            "source": "memorymesh-transfer",
        }

    return {
        "project": bundle.get("project"),
        "from_provider": bundle.get("from_provider"),
        "to_provider": key,
        "conversation_id": conv_id,
        "messages": normalized,
        "source": "memorymesh-transfer",
    }


def import_instructions(target: str, inbox_dir: Path) -> str:
    key = target.strip().lower()
    md = inbox_dir / "context.md"
    js = inbox_dir / "import.json"
    common = (
        f"MemoryMesh delivered transfer files to:\n"
        f"  - {md}\n"
        f"  - {js}\n\n"
    )
    if key == "cursor":
        return (
            common
            + "Cursor: open a new chat and paste the contents of context.md as your first message.\n"
            + "Optional: keep Cursor hooks installed so future sessions sync back into MemoryMesh.\n"
        )
    if key == "claude":
        return (
            common
            + "Claude Code: paste context.md into a new session, or import import.json via your export workflow.\n"
            + "Run `memorymesh install-native --target claude` if hooks are not installed.\n"
        )
    if key == "gemini":
        return (
            common
            + "Gemini: paste context.md into a new chat to continue with transferred context.\n"
        )
    if key == "continue":
        return (
            common
            + "Continue: paste context.md into the session input or add import.json to your session context.\n"
        )
    if key == "opencode":
        return (
            common
            + "OpenCode: paste context.md at session start; bridge plugin can ingest future exports.\n"
        )
    return common + f"{target}: paste context.md into a new chat to continue.\n"
