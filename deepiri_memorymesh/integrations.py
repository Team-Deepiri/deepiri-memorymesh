from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class IntegrationTarget:
    key: str
    label: str
    extension_hint: str
    hook_note: str


TARGETS: dict[str, IntegrationTarget] = {
    "cursor": IntegrationTarget(
        key="cursor",
        label="Cursor",
        extension_hint="VS Code-style extension task/command hook",
        hook_note="Use command/task hooks to call bridge script after chat export",
    ),
    "claude": IntegrationTarget(
        key="claude",
        label="Claude Code",
        extension_hint="shell wrapper/alias integration",
        hook_note="Wrap transcript export or periodic sync in shell profile",
    ),
    "gemini": IntegrationTarget(
        key="gemini",
        label="Gemini",
        extension_hint="CLI wrapper or extension callback",
        hook_note="Call bridge script with exported JSON file path",
    ),
    "opencode": IntegrationTarget(
        key="opencode",
        label="OpenCode",
        extension_hint="third-party plugin/extension command hook",
        hook_note="Use plugin hook to POST event payloads to MemoryMesh service",
    ),
    "continue": IntegrationTarget(
        key="continue",
        label="Continue.dev",
        extension_hint="custom command + post-action script",
        hook_note="Send session outputs to local service endpoint",
    ),
}


def list_targets() -> list[IntegrationTarget]:
    return [TARGETS[k] for k in sorted(TARGETS)]


def install_bridge_script(
    target: str,
    project: str,
    service_url: str = "http://127.0.0.1:8765",
) -> Path:
    key = target.strip().lower()
    if key not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    bin_dir = Path.home() / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / f"memorymesh-bridge-{key}"
    script_body = f"""#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: {script.name} <conversation_json_or_jsonl_file>" >&2
  exit 1
fi

FILE_PATH="$1"
SERVICE_URL="${{MEMORYMESH_URL:-{service_url}}}"
PROJECT="${{MEMORYMESH_PROJECT:-{project}}}"

curl -sS -X POST "$SERVICE_URL/ingest" \\
  -H "Content-Type: application/json" \\
  -d "{{\\"provider\\":\\"{key}\\",\\"project\\":\\"$PROJECT\\",\\"file_path\\":\\"$FILE_PATH\\"}}" >/dev/null

echo "memorymesh sync ok: provider={key} file=$FILE_PATH"
"""
    script.write_text(script_body, encoding="utf-8")
    script.chmod(0o755)
    return script


def write_integration_template(target: str, project: str) -> Path:
    key = target.strip().lower()
    if key not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    cfg_dir = Path.home() / ".config" / "deepiri-memorymesh" / "integrations"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"{key}.integration.json"
    bridge = str((Path.home() / ".local" / "bin" / f"memorymesh-bridge-{key}").expanduser())
    payload = (
        "{\n"
        f'  "target": "{key}",\n'
        f'  "project": "{project}",\n'
        f'  "bridge_command": "{bridge} /path/to/export.json",\n'
        '  "service_url": "${MEMORYMESH_URL:-http://127.0.0.1:8765}",\n'
        '  "hook_example": "Run bridge command as a post-export hook in your extension/plugin"\n'
        "}\n"
    )
    path.write_text(payload, encoding="utf-8")
    return path


def write_hook_snippets(project: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    cursor_tasks = output_dir / "cursor.tasks.json"
    cursor_tasks.write_text(
        "{\n"
        '  "version": "2.0.0",\n'
        '  "tasks": [\n'
        "    {\n"
        '      "label": "memorymesh-sync-cursor-export",\n'
        '      "type": "shell",\n'
        '      "command": "memorymesh-bridge-cursor ${input:conversationExport}",\n'
        '      "problemMatcher": []\n'
        "    }\n"
        "  ],\n"
        '  "inputs": [\n'
        "    {\n"
        '      "id": "conversationExport",\n'
        '      "type": "promptString",\n'
        '      "description": "Path to exported cursor conversation JSON/JSONL"\n'
        "    }\n"
        "  ]\n"
        "}\n",
        encoding="utf-8",
    )
    files.append(cursor_tasks)

    opencode_hook = output_dir / "opencode.hook.json"
    opencode_hook.write_text(
        "{\n"
        '  "name": "memorymesh-opencode-sync",\n'
        '  "event": "conversation_exported",\n'
        '  "run": "memorymesh-bridge-opencode ${conversation_file}",\n'
        f'  "env": {{"MEMORYMESH_PROJECT": "{project}"}}\n'
        "}\n",
        encoding="utf-8",
    )
    files.append(opencode_hook)

    continue_snippet = output_dir / "continue.command.json"
    continue_snippet.write_text(
        "{\n"
        '  "commands": [\n'
        "    {\n"
        '      "name": "MemoryMesh Sync",\n'
        '      "command": "memorymesh-bridge-continue ${input_file}"\n'
        "    }\n"
        "  ]\n"
        "}\n",
        encoding="utf-8",
    )
    files.append(continue_snippet)

    claude_alias = output_dir / "claude.alias.sh"
    claude_alias.write_text(
        "# Add to ~/.bashrc or ~/.zshrc\n"
        "alias claude-sync='memorymesh-bridge-claude'\n"
        f"export MEMORYMESH_PROJECT='{project}'\n",
        encoding="utf-8",
    )
    files.append(claude_alias)

    gemini_alias = output_dir / "gemini.alias.sh"
    gemini_alias.write_text(
        "# Add to ~/.bashrc or ~/.zshrc\n"
        "alias gemini-sync='memorymesh-bridge-gemini'\n"
        f"export MEMORYMESH_PROJECT='{project}'\n",
        encoding="utf-8",
    )
    files.append(gemini_alias)

    return files
