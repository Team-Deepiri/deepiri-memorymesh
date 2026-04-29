from __future__ import annotations

from dataclasses import dataclass
import json
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


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return path


def _append_command_hook(config: dict, event: str, command: str) -> dict:
    hooks = config.setdefault("hooks", {})
    event_hooks = hooks.setdefault(event, [])
    entry = {
        "matcher": ".*",
        "hooks": [{"type": "command", "command": command}],
    }
    if entry not in event_hooks:
        event_hooks.append(entry)
    return config


def install_bridge_script(
    target: str,
    project: str,
    service_url: str = "http://127.0.0.1:8765",
) -> Path:
    key = target.strip().lower()
    if key not in TARGETS and key != "jsonl":
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


def install_hook_script(
    target: str,
    project: str,
    service_url: str = "http://127.0.0.1:8765",
) -> Path:
    key = target.strip().lower()
    script = Path.home() / ".local" / "bin" / f"memorymesh-hook-{key}"
    script.parent.mkdir(parents=True, exist_ok=True)
    body = f"""#!/usr/bin/env bash
set -euo pipefail
SERVICE_URL="${{MEMORYMESH_URL:-{service_url}}}"
PROJECT="${{MEMORYMESH_PROJECT:-{project}}}"
PROVIDER="{key}"
INPUT="$(cat)"
TRANSCRIPT_PATH="$(INPUT_JSON="$INPUT" python3 -c 'import json,os; raw=os.environ.get("INPUT_JSON",""); 
try:
    obj=json.loads(raw) if raw.strip() else {{}}
except Exception:
    obj={{}}
print(obj.get("transcript_path",""))')"
if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
  curl -sS -X POST "$SERVICE_URL/ingest" \\
    -H "Content-Type: application/json" \\
    -d "{{\\"provider\\":\\"$PROVIDER\\",\\"project\\":\\"$PROJECT\\",\\"file_path\\":\\"$TRANSCRIPT_PATH\\"}}" >/dev/null || true
else
  curl -sS -X POST "$SERVICE_URL/ingest" \\
    -H "Content-Type: application/json" \\
    -d "{{\\"provider\\":\\"$PROVIDER\\",\\"project\\":\\"$PROJECT\\",\\"file_path\\":\\"$HOME/.claude/history.jsonl\\"}}" >/dev/null || true
fi
"""
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return script


def install_native_integration(
    target: str,
    project: str,
    service_url: str = "http://127.0.0.1:8765",
) -> list[Path]:
    key = target.strip().lower()
    if key not in TARGETS and key != "aider":
        raise ValueError(f"Unknown target: {target}")

    created: list[Path] = []
    bridge = install_bridge_script(key if key != "aider" else "jsonl", project, service_url)
    created.append(bridge)
    hook_script = install_hook_script(key if key != "aider" else "jsonl", project, service_url)
    created.append(hook_script)

    # Claude Code: ~/.claude/settings.json SessionEnd hook
    if key == "claude":
        settings_path = Path.home() / ".claude" / "settings.json"
        cfg = _load_json(settings_path)
        cfg = _append_command_hook(cfg, "SessionEnd", str(hook_script))
        created.append(_save_json(settings_path, cfg))
        return created

    # Cursor: ~/.cursor/hooks.json stop/sessionEnd style hook
    if key == "cursor":
        hooks_path = Path.home() / ".cursor" / "hooks.json"
        cfg = _load_json(hooks_path)
        cfg["version"] = cfg.get("version", 1)
        hooks = cfg.setdefault("hooks", {})
        stop_hooks = hooks.setdefault("stop", [])
        cursor_entry = {"command": str(hook_script)}
        if cursor_entry not in stop_hooks:
            stop_hooks.append(cursor_entry)
        created.append(_save_json(hooks_path, cfg))
        return created

    # Gemini CLI: ~/.gemini/settings.json SessionEnd command hook
    if key == "gemini":
        settings_path = Path.home() / ".gemini" / "settings.json"
        cfg = _load_json(settings_path)
        cfg = _append_command_hook(cfg, "SessionEnd", str(hook_script))
        created.append(_save_json(settings_path, cfg))
        return created

    # Continue CLI: ~/.continue/settings.json SessionEnd hook
    if key == "continue":
        settings_path = Path.home() / ".continue" / "settings.json"
        cfg = _load_json(settings_path)
        cfg = _append_command_hook(cfg, "SessionEnd", str(hook_script))
        created.append(_save_json(settings_path, cfg))
        return created

    # OpenCode: plugin file in ~/.config/opencode/plugin/
    if key == "opencode":
        plugin_dir = Path.home() / ".config" / "opencode" / "plugin"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        plugin_path = plugin_dir / "memorymesh.ts"
        plugin_path.write_text(
            "import type { Plugin } from \"@opencode-ai/plugin\"\n\n"
            "export const MemoryMeshPlugin: Plugin = async () => {\n"
            "  return {\n"
            "    event: async ({ event }) => {\n"
            "      if (event.type !== \"session.idle\") return\n"
            "      const cmd = `${process.env.HOME}/.local/bin/memorymesh-bridge-opencode ${process.env.HOME}/.claude/history.jsonl`\n"
            "      await Bun.$`${['bash','-lc',cmd]}`\n"
            "    },\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        created.append(plugin_path)
        return created

    # Aider: wrapper launcher with history file + auto ingest on exit
    if key == "aider":
        wrapper = Path.home() / ".local" / "bin" / "aider-memorymesh"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "PROJECT=\"${MEMORYMESH_PROJECT:-" + project + "}\"\n"
            "STAMP=$(date +%Y%m%d_%H%M%S)\n"
            "OUT=\"${HOME}/.aider.history/${STAMP}.chat.md\"\n"
            "mkdir -p \"${HOME}/.aider.history\"\n"
            "aider --chat-history-file \"$OUT\" \"$@\"\n"
            "memorymesh ingest --provider aider --project \"$PROJECT\" --file \"$OUT\" || true\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        created.append(wrapper)
        return created

    return created


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
