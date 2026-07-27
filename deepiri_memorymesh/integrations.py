from __future__ import annotations

"""Bridge/hook/native installers for MemoryMesh integrations.

Re-run ``memorymesh install-integration`` / ``install-native`` after upgrades
to regenerate scripts (existing home-directory scripts are not rewritten
automatically).
"""

from dataclasses import dataclass
import json
import shlex
from pathlib import Path

from .providers.registry import (
    allows_native_integration,
    get_provider,
    native_integration_provider_names,
    normalize_provider_key,
)


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
        extension_hint="native OpenCode TypeScript plugin",
        hook_note="Installs ~/.config/opencode/plugins/memorymesh.ts",
    ),
    "continue": IntegrationTarget(
        key="continue",
        label="Continue.dev",
        extension_hint="custom command + post-action script",
        hook_note="Send session outputs to local service endpoint",
    ),
}

# Event JSON keys inspected for a transcript/session file path.
_PATH_KEYS_BY_PROVIDER: dict[str, tuple[str, ...]] = {
    "claude": ("transcript_path", "file_path", "path"),
    "cursor": ("transcript_path", "file_path", "path"),
    "gemini": ("transcript_path", "file_path", "path"),
    "opencode": ("transcript_path", "file_path", "path"),
    "continue": ("transcript_path", "file_path", "path"),
    "jsonl": ("transcript_path", "file_path", "path"),
    "aider": ("transcript_path", "file_path", "path"),
}

_CLAUDE_HISTORY_FALLBACK = "~/.claude/history.jsonl"

# Ownership marker key stamped onto hook-config entries we create, so
# uninstall can remove exactly (and only) what a given installation added.
OWNERSHIP_MARKER_KEY = "memorymesh_ownership"


def list_targets() -> list[IntegrationTarget]:
    """Integration targets derived from the provider capability registry."""
    out: list[IntegrationTarget] = []
    for name in native_integration_provider_names():
        if name in TARGETS:
            out.append(TARGETS[name])
        else:
            # Native with integration support but no rich metadata yet.
            out.append(
                IntegrationTarget(
                    key=name,
                    label=name,
                    extension_hint="native integration",
                    hook_note="Installed via memorymesh install-native",
                )
            )
    return out


def _assert_integration_allowed(target: str) -> str:
    key = normalize_provider_key(target)
    if not allows_native_integration(key):
        cap = get_provider(key)
        raise ValueError(
            f"Unsupported or non-integrable provider {target!r} "
            f"(kind={cap.parser_kind}, integration={cap.integration_support})"
        )
    return key


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_json(path: Path, payload: dict | list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


def _same_owner(candidate: object, marker: dict[str, str]) -> bool:
    return (
        isinstance(candidate, dict)
        and candidate.get("installation_id") == marker.get("installation_id")
        and candidate.get("product") == marker.get("product")
    )


def _append_owned_command_hook(
    config: dict, event: str, command: str, marker: dict[str, str]
) -> dict:
    """Add a matcher-style command hook (Claude/Gemini/Continue) carrying an
    ownership marker so a later transactional uninstall can remove exactly
    this installation's entry without touching unrelated hooks.

    Re-running install for the same ``installation_id`` replaces (rather than
    duplicates) its own prior entry for this event.
    """
    hooks = config.setdefault("hooks", {})
    event_hooks = hooks.setdefault(event, [])
    event_hooks[:] = [
        entry
        for entry in event_hooks
        if not (isinstance(entry, dict) and _same_owner(entry.get(OWNERSHIP_MARKER_KEY), marker))
    ]
    entry = {
        "matcher": ".*",
        "hooks": [{"type": "command", "command": command}],
        OWNERSHIP_MARKER_KEY: dict(marker),
    }
    event_hooks.append(entry)
    return config


def _append_owned_stop_hook(config: dict, command: str, marker: dict[str, str]) -> dict:
    """Add a Cursor ``hooks.stop`` entry carrying an ownership marker."""
    cfg = config
    cfg["version"] = cfg.get("version", 1)
    hooks = cfg.setdefault("hooks", {})
    stop_hooks = hooks.setdefault("stop", [])
    stop_hooks[:] = [
        entry
        for entry in stop_hooks
        if not (isinstance(entry, dict) and _same_owner(entry.get(OWNERSHIP_MARKER_KEY), marker))
    ]
    stop_hooks.append({"command": command, OWNERSHIP_MARKER_KEY: dict(marker)})
    return cfg


def build_ingest_payload(provider: str, project: str, file_path: str) -> str:
    """Serialize an ingest JSON body with correct escaping (stdlib only)."""
    return json.dumps(
        {"provider": provider, "project": project, "file_path": file_path},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _shell_default_assignments(project: str, service_url: str) -> list[str]:
    """Embed caller values as shell-quoted literals (never raw interpolation)."""
    return [
        f"DEFAULT_PROJECT={shlex.quote(project)}",
        f"DEFAULT_SERVICE_URL={shlex.quote(service_url)}",
        'PROJECT="${MEMORYMESH_PROJECT:-$DEFAULT_PROJECT}"',
        'SERVICE_URL="${MEMORYMESH_URL:-$DEFAULT_SERVICE_URL}"',
    ]


def _bearer_token_env_fragment() -> str:
    """Bash fragment: resolve an optional bearer token at RUNTIME only.

    Reads ``MEMORYMESH_TOKEN`` directly, or a token stored in the file path
    given by ``MEMORYMESH_TOKEN_FILE``. Never embeds a literal token in the
    generated script. Warns (non-fatally) if the token file looks readable
    by group/other. When neither is set, ``AUTH_TOKEN`` stays empty and the
    request is sent without an ``Authorization`` header — the service then
    correctly responds 401 if it requires auth.

    Deliberately avoids bash arrays for the (optional) header: under
    ``set -u``, referencing an empty array with ``"${arr[@]}"`` is treated as
    an unbound variable on bash 3.2 (macOS's default ``/bin/bash``), so the
    caller branches on ``[ -n "$AUTH_TOKEN" ]`` instead of building an args array.
    """
    return (
        'AUTH_TOKEN=""\n'
        'if [ -n "${MEMORYMESH_TOKEN:-}" ]; then\n'
        '  AUTH_TOKEN="$MEMORYMESH_TOKEN"\n'
        'elif [ -n "${MEMORYMESH_TOKEN_FILE:-}" ] && [ -f "${MEMORYMESH_TOKEN_FILE}" ]; then\n'
        "  MM_TOKEN_FILE_PERM=\"$(stat -c '%a' \"${MEMORYMESH_TOKEN_FILE}\" 2>/dev/null "
        "|| stat -f '%Lp' \"${MEMORYMESH_TOKEN_FILE}\" 2>/dev/null || echo '')\"\n"
        '  if [ -n "$MM_TOKEN_FILE_PERM" ] && [ "${MM_TOKEN_FILE_PERM: -2}" != "00" ]; then\n'
        '    echo "memorymesh: warning: MEMORYMESH_TOKEN_FILE is group/other-accessible '
        '(mode $MM_TOKEN_FILE_PERM); consider chmod 600" >&2\n'
        "  fi\n"
        '  AUTH_TOKEN="$(cat "${MEMORYMESH_TOKEN_FILE}")"\n'
        "fi"
    )


# Python program (run via ``python3 -c "$(cat <<'HEREDOC' ... HEREDOC)"`` so no
# shell quoting/escaping of the token ever occurs) that emits a curl
# ``--config`` file body from the ``AUTH_TOKEN`` env var. Always includes
# ``Content-Type``; includes ``Authorization: Bearer`` only when a token is
# present. curl's config-file syntax requires the header value to be a
# double-quoted string, so a literal backslash or double-quote inside the
# token is escaped before being embedded (raw string: no extra escaping by
# *this* source file is needed — the bytes below are exactly what curl needs).
_CURL_CONFIG_FROM_ENV_PY = r"""import os
lines = ['header = "Content-Type: application/json"']
token = os.environ.get("AUTH_TOKEN", "")
if token:
    esc = token.replace("\\", "\\\\").replace('"', '\\"')
    lines.append('header = "Authorization: Bearer ' + esc + '"')
print("\n".join(lines))
"""


def _curl_config_from_env_python() -> str:
    """Python program: emit a curl ``--config`` file body from ``AUTH_TOKEN``."""
    return _CURL_CONFIG_FROM_ENV_PY


def _curl_ingest_from_env_fragment() -> str:
    """Bash fragment: build JSON via python3 json.dumps and POST via curl stdin.

    Expects env vars SERVICE_URL, PROVIDER, PROJECT, FILE_PATH. Also resolves
    an optional bearer token at runtime from MEMORYMESH_TOKEN or
    MEMORYMESH_TOKEN_FILE (see :func:`_bearer_token_env_fragment`) and sends
    it as ``Authorization: Bearer`` — never as a query-string parameter, never
    embedded as a literal in this generated script, and (critically) never
    passed on curl's command line/argv, where it would be visible to any
    other local user via ``ps``. Instead a mode-0600 temporary curl
    ``--config`` file is written containing the headers, referenced via
    ``--config``, and removed on exit via ``trap``. If no token is available,
    the config file omits the ``Authorization`` header and the service
    returns 401 when it requires auth (which is the correct behavior).
    Uses portable ``curl --fail`` so HTTP errors yield a nonzero curl exit status
    without requiring a minimum curl version for ``--fail-with-body``.
    """
    build_payload = (
        "python3 -c 'import json,os,sys; "
        "sys.stdout.write(json.dumps("
        '{"provider":os.environ["PROVIDER"],'
        '"project":os.environ["PROJECT"],'
        '"file_path":os.environ["FILE_PATH"]},'
        "ensure_ascii=False))'"
    )
    config_writer_py = _curl_config_from_env_python()
    return (
        f"{_bearer_token_env_fragment()}\n"
        'MM_CURL_CFG="$(mktemp "${TMPDIR:-/tmp}/memorymesh-curl.XXXXXX")"\n'
        'chmod 600 "$MM_CURL_CFG"\n'
        "trap 'rm -f \"$MM_CURL_CFG\"' EXIT\n"
        'AUTH_TOKEN="$AUTH_TOKEN" python3 -c "$(cat <<\'MM_CURLCFG_PY\'\n'
        f"{config_writer_py.rstrip(chr(10))}\n"
        "MM_CURLCFG_PY\n"
        ')" > "$MM_CURL_CFG"\n'
        f"{build_payload} \\\n"
        '  | curl -sS --fail -X POST "$SERVICE_URL/ingest" \\\n'
        '    --config "$MM_CURL_CFG" \\\n'
        "    --data-binary @-"
    )


def _hook_resolve_path_python(provider: str) -> str:
    """Python program that prints a usable transcript path from INPUT_JSON."""
    keys = list(_PATH_KEYS_BY_PROVIDER.get(provider, ("transcript_path", "file_path", "path")))
    lines = [
        "import json, os",
        "raw = os.environ.get('INPUT_JSON', '')",
        "try:",
        "    obj = json.loads(raw) if raw.strip() else {}",
        "except Exception:",
        "    obj = {}",
        f"keys = {json.dumps(keys)}",
        "path = ''",
        "for k in keys:",
        "    v = obj.get(k) if isinstance(obj, dict) else None",
        "    if isinstance(v, str) and v.strip():",
        "        path = v.strip()",
        "        break",
        "if path and os.path.isfile(path):",
        "    print(path)",
        "    raise SystemExit(0)",
    ]
    if provider == "claude":
        lines.extend(
            [
                f"fallback = os.path.expanduser({_CLAUDE_HISTORY_FALLBACK!r})",
                "if os.path.isfile(fallback):",
                "    print(fallback)",
                "    raise SystemExit(0)",
            ]
        )
    lines.append("print('')")
    return "\n".join(lines) + "\n"


def _stamp_ownership_comment(body: str, installation_id: str | None, provider: str) -> str:
    """Insert a ``# memorymesh-generated ...`` marker line right after the shebang.

    Pure string transform; used by render helpers so a caller (e.g. the
    transactional installer) can trace a generated file back to the
    installation that created it, without any filesystem access here.
    """
    if not installation_id:
        return body
    marker = f"# memorymesh-generated installation_id={installation_id} provider={provider}\n"
    if body.startswith("#!"):
        nl = body.find("\n")
        if nl >= 0:
            return body[: nl + 1] + marker + body[nl + 1 :]
        return body + "\n" + marker
    return marker + body


def render_bridge_script(
    target: str,
    project: str,
    service_url: str = "http://127.0.0.1:8765",
    *,
    installation_id: str | None = None,
) -> tuple[Path, str]:
    """Pure: compute the bridge script's destination path and full text.

    Performs no filesystem access/writes; callers decide if/how to persist.
    """
    key = _assert_integration_allowed(target)
    script = Path.home() / ".local" / "bin" / f"memorymesh-bridge-{key}"
    curl_frag = _curl_ingest_from_env_fragment()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'if [ "$#" -lt 1 ]; then',
        f'  echo "usage: {script.name} <conversation_json_or_jsonl_file>" >&2',
        "  exit 1",
        "fi",
        "",
        'FILE_PATH="$1"',
        *_shell_default_assignments(project, service_url),
        f'PROVIDER="{key}"',
        "export SERVICE_URL PROVIDER PROJECT FILE_PATH",
        "",
        # Explicit bridge: nonzero exit on HTTP/network failure; no false success.
        "set +e",
        f"{curl_frag} >/dev/null",
        "INGEST_RC=$?",
        "set -e",
        'if [ "$INGEST_RC" -ne 0 ]; then',
        f'  echo "memorymesh-bridge-{key}: ingest failed" >&2',
        "  exit 1",
        "fi",
        "",
        f'echo "memorymesh sync ok: provider={key} file=$FILE_PATH"',
        "",
    ]
    body = _stamp_ownership_comment("\n".join(lines), installation_id, key)
    return script, body


def install_bridge_script(
    target: str,
    project: str,
    service_url: str = "http://127.0.0.1:8765",
) -> Path:
    """Thin write wrapper around :func:`render_bridge_script` (back-compat)."""
    script, body = render_bridge_script(target, project, service_url)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return script


def render_hook_script(
    target: str,
    project: str,
    service_url: str = "http://127.0.0.1:8765",
    *,
    installation_id: str | None = None,
) -> tuple[Path, str]:
    """Pure: compute the hook script's destination path and full text."""
    key = _assert_integration_allowed(target)
    script = Path.home() / ".local" / "bin" / f"memorymesh-hook-{key}"
    curl_frag = _curl_ingest_from_env_fragment()
    resolver = _hook_resolve_path_python(key)
    parts = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        *_shell_default_assignments(project, service_url),
        f'PROVIDER="{key}"',
        'INPUT="$(cat)"',
        "FILE_PATH=\"$(",
        '  INPUT_JSON="$INPUT" python3 -c "$(cat <<\'MM_HOOK_PY\'',
        resolver.rstrip("\n"),
        "MM_HOOK_PY",
        ')"',
        ')"',
        'if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then',
        f'  echo "memorymesh-hook-{key}: no usable transcript path; skipping ingest" >&2',
        "  exit 0",
        "fi",
        "export SERVICE_URL PROVIDER PROJECT FILE_PATH",
        # Soft-fail by default (do not break IDE hosts); MEMORYMESH_STRICT=1 propagates.
        "set +e",
        f"{curl_frag} >/dev/null",
        "INGEST_RC=$?",
        "set -e",
        'if [ "$INGEST_RC" -ne 0 ]; then',
        f'  echo "memorymesh-hook-{key}: ingest failed" >&2',
        '  if [ "${MEMORYMESH_STRICT:-0}" = "1" ]; then',
        "    exit 1",
        "  fi",
        "  exit 0",
        "fi",
        "",
    ]
    body = _stamp_ownership_comment("\n".join(parts), installation_id, key)
    return script, body


def install_hook_script(
    target: str,
    project: str,
    service_url: str = "http://127.0.0.1:8765",
) -> Path:
    """Thin write wrapper around :func:`render_hook_script` (back-compat)."""
    script, body = render_hook_script(target, project, service_url)
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return script


def _opencode_plugin_source(project: str, service_url: str) -> str:
    """Generate OpenCode plugin TypeScript.

    Handles official ``session.idle`` and ``session.status`` (idle) events.
    Uses ``client.session.messages`` and posts inline JSON to ``/ingest``.
    Deduplicates concurrent/legacy idle notifications for one transition while
    allowing later genuine idle cycles and retries after failure.
    """
    project_js = json.dumps(project)
    service_url_js = json.dumps(service_url)
    return f"""import type {{ Plugin }} from "@opencode-ai/plugin"

// Resolves an optional bearer token at RUNTIME only — never embedded as a
// literal here. Reads MEMORYMESH_TOKEN directly, or a token stored in the
// file path given by MEMORYMESH_TOKEN_FILE. Returns undefined when neither
// is set, in which case requests are sent without an Authorization header
// and the service correctly responds 401 if it requires auth.
//
// The "node:fs" specifier is kept in a variable (not a string literal) so
// bundlers targeting non-Node platforms (e.g. esbuild --platform=neutral)
// pass this dynamic import through untouched instead of trying to resolve
// a Node builtin at bundle time; Node itself resolves it fine at runtime.
async function resolveAuthToken(): Promise<string | undefined> {{
  const direct = process.env.MEMORYMESH_TOKEN
  if (direct && direct.trim()) return direct.trim()
  const tokenFile = process.env.MEMORYMESH_TOKEN_FILE
  if (tokenFile && tokenFile.trim()) {{
    try {{
      const nodeFsSpecifier = "node:fs"
      const fs = await import(nodeFsSpecifier)
      return String(fs.readFileSync(tokenFile, "utf-8")).trim()
    }} catch (err) {{
      const detail = err instanceof Error ? err.message : String(err)
      console.error(`memorymesh: could not read MEMORYMESH_TOKEN_FILE (${{detail}})`)
    }}
  }}
  return undefined
}}

function unwrapMessages(raw: unknown): Array<{{ info?: Record<string, unknown>; parts?: unknown[] }}> {{
  if (Array.isArray(raw)) return raw as Array<{{ info?: Record<string, unknown>; parts?: unknown[] }}>
  if (raw && typeof raw === "object" && Array.isArray((raw as {{ data?: unknown }}).data)) {{
    return (raw as {{ data: Array<{{ info?: Record<string, unknown>; parts?: unknown[] }}> }}).data
  }}
  return []
}}

function textFromParts(parts: unknown[]): string {{
  const chunks: string[] = []
  for (const part of parts) {{
    if (!part || typeof part !== "object") continue
    const p = part as {{ type?: unknown; text?: unknown }}
    if (p.type === "text" && typeof p.text === "string" && p.text.trim()) {{
      chunks.push(p.text)
    }}
  }}
  return chunks.join("\\n")
}}

function timestampFromInfo(info: Record<string, unknown>): string | undefined {{
  const time = info.time
  if (time && typeof time === "object") {{
    const created = (time as {{ created?: unknown }}).created
    if (typeof created === "number" && Number.isFinite(created)) {{
      return new Date(created).toISOString()
    }}
    if (typeof created === "string" && created.trim()) return created
  }}
  return undefined
}}

function toConversation(sessionID: string, rows: Array<{{ info?: Record<string, unknown>; parts?: unknown[] }}>) {{
  const messages: Array<Record<string, unknown>> = []
  for (const row of rows) {{
    const info = (row.info && typeof row.info === "object") ? row.info : {{}}
    const parts = Array.isArray(row.parts) ? row.parts : []
    const content = textFromParts(parts)
    if (!content) continue
    const role = typeof info.role === "string" && info.role.trim() ? info.role : "unknown"
    const messageID = typeof info.id === "string" ? info.id : undefined
    const msg: Record<string, unknown> = {{
      role,
      content,
      metadata: {{
        source: "opencode",
        session_id: sessionID,
        ...(messageID ? {{ message_id: messageID }} : {{}}),
      }},
    }}
    const ts = timestampFromInfo(info)
    if (ts) msg.timestamp = ts
    messages.push(msg)
  }}
  return {{ conversation_id: sessionID, messages }}
}}

export const MemoryMeshPlugin: Plugin = async ({{ client }}) => {{
  const defaultUrl = {service_url_js}
  const defaultProject = {project_js}
  const inFlight = new Set<string>()
  const completedIdle = new Set<string>()

  const ingestSession = async (sessionID: string) => {{
    if (inFlight.has(sessionID) || completedIdle.has(sessionID)) return
    inFlight.add(sessionID)
    try {{
      const raw = await client.session.messages({{ path: {{ id: sessionID }} }})
      const conversation = toConversation(sessionID, unwrapMessages(raw))
      const serviceUrl = process.env.MEMORYMESH_URL || defaultUrl
      const project = process.env.MEMORYMESH_PROJECT || defaultProject
      const payload = {{
        provider: "opencode",
        project,
        conversation,
      }}
      const headers: Record<string, string> = {{ "Content-Type": "application/json" }}
      const token = await resolveAuthToken()
      if (token) headers["Authorization"] = `Bearer ${{token}}`
      const res = await fetch(`${{serviceUrl}}/ingest`, {{
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      }})
      if (!res.ok) {{
        throw new Error(`ingest HTTP ${{res.status}}`)
      }}
      completedIdle.add(sessionID)
    }} catch (err) {{
      const detail = err instanceof Error ? err.message : String(err)
      console.error(`memorymesh: OpenCode session ingest failed; skipping (${{detail}})`)
    }} finally {{
      inFlight.delete(sessionID)
    }}
  }}

  return {{
    event: async ({{ event }}) => {{
      if (event.type === "session.status") {{
        const props = (event.properties ?? {{}}) as {{
          sessionID?: unknown
          status?: {{ type?: unknown }}
        }}
        const sessionID = typeof props.sessionID === "string" ? props.sessionID : ""
        const statusType = props.status && typeof props.status === "object" ? props.status.type : undefined
        if (statusType !== "idle") {{
          if (sessionID) completedIdle.delete(sessionID)
          return
        }}
        if (!sessionID.trim()) {{
          console.error("memorymesh: OpenCode session.status idle missing sessionID; skipping ingest")
          return
        }}
        await ingestSession(sessionID)
        return
      }}

      if (event.type !== "session.idle") return
      const sessionID = (event.properties as {{ sessionID?: unknown }} | undefined)?.sessionID
      if (typeof sessionID !== "string" || !sessionID.trim()) {{
        console.error("memorymesh: OpenCode session.idle missing sessionID; skipping ingest")
        return
      }}
      await ingestSession(sessionID)
    }},
  }}
}}
"""


def render_opencode_plugin(
    project: str,
    service_url: str = "http://127.0.0.1:8765",
    *,
    installation_id: str | None = None,
) -> tuple[Path, str]:
    """Pure: compute the OpenCode plugin's destination path and full text."""
    plugin_path = Path.home() / ".config" / "opencode" / "plugins" / "memorymesh.ts"
    source = _opencode_plugin_source(project=project, service_url=service_url)
    if installation_id:
        header = (
            f"// memorymesh-generated installation_id={installation_id} provider=opencode\n"
        )
        source = header + source
    return plugin_path, source


def render_aider_wrapper(
    project: str,
    *,
    installation_id: str | None = None,
) -> tuple[Path, str]:
    """Pure: compute the aider wrapper's destination path and full text.

    Preserves Aider's exit code. Ingest failure is reported; default soft,
    ``MEMORYMESH_STRICT=1`` fails only when Aider itself succeeded.
    """
    wrapper = Path.home() / ".local" / "bin" / "aider-memorymesh"
    quoted_project = shlex.quote(project)
    body = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"DEFAULT_PROJECT={quoted_project}\n"
        'PROJECT="${MEMORYMESH_PROJECT:-$DEFAULT_PROJECT}"\n'
        "STAMP=$(date +%Y%m%d_%H%M%S)\n"
        'OUT="${HOME}/.aider.history/${STAMP}.chat.md"\n'
        'mkdir -p "${HOME}/.aider.history"\n'
        "set +e\n"
        'aider --chat-history-file "$OUT" -- "$@"\n'
        "AIDER_RC=$?\n"
        "set -e\n"
        "set +e\n"
        'memorymesh ingest --provider aider --project "$PROJECT" --file "$OUT"\n'
        "INGEST_RC=$?\n"
        "set -e\n"
        'if [ "$INGEST_RC" -ne 0 ]; then\n'
        '  echo "aider-memorymesh: Memory Mesh ingest failed (exit $INGEST_RC)" >&2\n'
        '  if [ "${MEMORYMESH_STRICT:-0}" = "1" ] && [ "$AIDER_RC" -eq 0 ]; then\n'
        "    exit 1\n"
        "  fi\n"
        "fi\n"
        'exit "$AIDER_RC"\n'
    )
    body = _stamp_ownership_comment(body, installation_id, "aider")
    return wrapper, body


def install_push_script(target: str) -> Path:
    key = target.strip().lower()
    bin_dir = Path.home() / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / f"memorymesh-push-{key}"
    script_body = f"""#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: {script.name} <transfer_bundle.json>" >&2
  exit 1
fi

FILE_PATH="$1"
exec python3 -m deepiri_memorymesh.cli transfer-deliver --bundle "$FILE_PATH" --to {key}
"""
    script.write_text(script_body, encoding="utf-8")
    script.chmod(0o755)
    return script


def install_native_integration(
    target: str,
    project: str,
    service_url: str = "http://127.0.0.1:8765",
) -> list[Path]:
    key = _assert_integration_allowed(target)
    # Native install is for native parsers; jsonl may use bridge helpers only.
    if get_provider(key).parser_kind == "generic-explicit":
        raise ValueError(
            f"Provider {target!r} is generic-explicit; use install-integration/bridge, "
            "not install-native"
        )

    created: list[Path] = []
    created.append(install_push_script(key))

    # OpenCode: native SDK plugin only (no unused transcript bridge/hook).
    if key == "opencode":
        plugin_path, source = render_opencode_plugin(project=project, service_url=service_url)
        plugin_path.parent.mkdir(parents=True, exist_ok=True)
        plugin_path.write_text(source, encoding="utf-8")
        created.append(plugin_path)
        return created

    bridge = install_bridge_script(key if key != "aider" else "jsonl", project, service_url)
    created.append(bridge)
    hook_script = install_hook_script(key if key != "aider" else "jsonl", project, service_url)
    created.append(hook_script)

    if key == "claude":
        settings_path = Path.home() / ".claude" / "settings.json"
        cfg = _load_json(settings_path)
        cfg = _append_command_hook(cfg, "SessionEnd", str(hook_script))
        created.append(_save_json(settings_path, cfg))
        return created

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

    if key == "gemini":
        settings_path = Path.home() / ".gemini" / "settings.json"
        cfg = _load_json(settings_path)
        cfg = _append_command_hook(cfg, "SessionEnd", str(hook_script))
        created.append(_save_json(settings_path, cfg))
        return created

    if key == "continue":
        settings_path = Path.home() / ".continue" / "settings.json"
        cfg = _load_json(settings_path)
        cfg = _append_command_hook(cfg, "SessionEnd", str(hook_script))
        created.append(_save_json(settings_path, cfg))
        return created

    if key == "aider":
        wrapper, body = render_aider_wrapper(project)
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(body, encoding="utf-8")
        wrapper.chmod(0o755)
        created.append(wrapper)
        return created

    return created


def write_integration_template(target: str, project: str) -> Path:
    key = _assert_integration_allowed(target)
    cfg_dir = Path.home() / ".config" / "deepiri-memorymesh" / "integrations"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"{key}.integration.json"
    bridge = str((Path.home() / ".local" / "bin" / f"memorymesh-bridge-{key}").expanduser())
    payload = {
        "target": key,
        "project": project,
        "bridge_command": f"{bridge} /path/to/export.json",
        "service_url": "${MEMORYMESH_URL:-http://127.0.0.1:8765}",
        "hook_example": "Run bridge command as a post-export hook in your extension/plugin",
    }
    if key == "opencode":
        payload = {
            "target": key,
            "project": project,
            "integration": "native-plugin",
            "plugin_path": str(Path.home() / ".config" / "opencode" / "plugins" / "memorymesh.ts"),
            "install": "memorymesh install-native --target opencode --project <project>",
            "service_url": "${MEMORYMESH_URL:-http://127.0.0.1:8765}",
            "events": ["session.idle", "session.status(idle)"],
            "hook_example": "OpenCode loads the plugin automatically from ~/.config/opencode/plugins/",
        }
    return _save_json(path, payload)


def write_hook_snippets(project: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    cursor_tasks = output_dir / "cursor.tasks.json"
    _save_json(
        cursor_tasks,
        {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "memorymesh-sync-cursor-export",
                    "type": "shell",
                    "command": "memorymesh-bridge-cursor ${input:conversationExport}",
                    "problemMatcher": [],
                }
            ],
            "inputs": [
                {
                    "id": "conversationExport",
                    "type": "promptString",
                    "description": "Path to exported cursor conversation JSON/JSONL",
                }
            ],
        },
    )
    files.append(cursor_tasks)

    opencode_plugin = output_dir / "opencode.plugin.json"
    _save_json(
        opencode_plugin,
        {
            "name": "memorymesh-opencode",
            "integration": "native-plugin",
            "plugin_path": "~/.config/opencode/plugins/memorymesh.ts",
            "install": f"memorymesh install-native --target opencode --project {project}",
            "project": project,
            "events": ["session.idle", "session.status(idle)"],
            "behavior": (
                "On idle, calls client.session.messages and POSTs an inline "
                "conversation payload to Memory Mesh /ingest. No transcript file bridge."
            ),
        },
    )
    files.append(opencode_plugin)

    continue_snippet = output_dir / "continue.command.json"
    _save_json(
        continue_snippet,
        {
            "commands": [
                {
                    "name": "MemoryMesh Sync",
                    "command": "memorymesh-bridge-continue ${input_file}",
                }
            ]
        },
    )
    files.append(continue_snippet)

    claude_alias = output_dir / "claude.alias.sh"
    claude_alias.write_text(
        "# Add to ~/.bashrc or ~/.zshrc\n"
        "alias claude-sync='memorymesh-bridge-claude'\n"
        f"export MEMORYMESH_PROJECT={shlex.quote(project)}\n",
        encoding="utf-8",
    )
    files.append(claude_alias)

    gemini_alias = output_dir / "gemini.alias.sh"
    gemini_alias.write_text(
        "# Add to ~/.bashrc or ~/.zshrc\n"
        "alias gemini-sync='memorymesh-bridge-gemini'\n"
        f"export MEMORYMESH_PROJECT={shlex.quote(project)}\n",
        encoding="utf-8",
    )
    files.append(gemini_alias)

    return files
