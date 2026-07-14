# Code App Integrations (Beyond CLI)

`deepiri-memorymesh` supports direct integration with code apps via:

- local HTTP service (`memorymesh serve`)
- per-app bridge scripts (`memorymesh-bridge-<target>`)
- native OpenCode plugin (`~/.config/opencode/plugins/memorymesh.ts`)
- extension/plugin hook templates

This allows syncing from tools even when they run through third-party extensions.

## 1) Start service

```bash
memorymesh auth token create --project deepiri --scope read --scope write \
  --token-file ~/.config/deepiri-memorymesh/tokens/deepiri.token
memorymesh serve --host 127.0.0.1 --port 8765
```

Project endpoints require `Authorization: Bearer <token>` by default. `/health` stays open.
Use `--auth-mode off` only for trusted local development (prints a warning).

API endpoints:

- `GET /health`
- `GET /stats?project=<name>` (read scope)
- `POST /ingest` (write scope; file path or raw conversation payload)
- `POST /query` (read scope)
- `POST /state/put` (write scope)
- `POST /state/get` (read scope)

## 2) Install integration bridge

```bash
memorymesh install-integration --target cursor --project deepiri
memorymesh install-native --target opencode --project deepiri \
  --token-file ~/.config/deepiri-memorymesh/tokens/deepiri.token
memorymesh install-native --target opencode --project deepiri --dry-run
memorymesh uninstall-native --target opencode
memorymesh integrations-mgmt status
memorymesh integrations-mgmt verify
```

`install-native` is transactional (T12): preflight, optional backup, atomic writes, manifest under
`~/.config/deepiri-memorymesh/integrations/`. Manifests store a **token-file path**, never token
content. Generated scripts read `MEMORYMESH_TOKEN` / `MEMORYMESH_TOKEN_FILE` at runtime.

`uninstall-native` removes only Memory Mesh-owned artifacts. Modified generated files are preserved
unless `--force` is passed (still never deletes unrelated paths).

This installs:

- executable bridge/hook scripts in `~/.local/bin/` (file-export providers)
- for OpenCode: the native plugin at `~/.config/opencode/plugins/memorymesh.ts`
- installation manifests in `~/.config/deepiri-memorymesh/integrations/`

Re-run `install-integration` / `install-native --update` after upgrading MemoryMesh to regenerate bridge and hook scripts when inputs change. Embedded shell defaults are written with `shlex.quote`. JSON ingest payloads are built with Python `json.dumps` and sent to curl via stdin.

### Failure policy (hooks / bridges)

- **Explicit bridges** (`memorymesh-bridge-<provider>`): nonzero exit on HTTP/network failure; prints a concise stderr diagnostic; does **not** print `sync ok` after failure. Uses portable `curl --fail`.
- **IDE/provider hooks** (`memorymesh-hook-<provider>`): report ingest failures on stderr. Default soft-failure exits `0` so host IDEs are not broken. Set `MEMORYMESH_STRICT=1` to propagate a nonzero exit.
- **Aider wrapper** (`aider-memorymesh`): always preserves Aider’s exit code. If Aider succeeds but ingest fails, prints a diagnostic; default keeps Aider’s success, `MEMORYMESH_STRICT=1` returns failure. An Aider failure is never replaced by an ingest result.
- **Transfer `--push`**: missing bridges and nonzero bridge exits are reported (CLI exits nonzero); success is not claimed when the push fails.

Generate ready-to-paste hook snippets:

```bash
memorymesh generate-hook-snippets --project deepiri --out-dir ./memorymesh-hooks
```

Generated files include:

- `cursor.tasks.json`
- `opencode.plugin.json` (documents the native plugin path and idle events)
- `continue.command.json`
- `claude.alias.sh`
- `gemini.alias.sh`

## 3) Hook your app/plugin

For file-export providers, use a post-export command:

```bash
memorymesh-bridge-cursor /path/to/export.json
```

### OpenCode (native plugin)

OpenCode integration is the TypeScript plugin under:

`~/.config/opencode/plugins/memorymesh.ts`

Install with:

```bash
memorymesh install-native --target opencode --project deepiri
```

On `session.status` (idle) and legacy `session.idle`, the plugin:

1. reads `event.properties.sessionID`
2. calls `client.session.messages({ path: { id: sessionID } })`
3. posts an inline conversation JSON body to Memory Mesh `/ingest`

No transcript-file bridge is required for OpenCode. Duplicate idle notifications for the same transition are deduplicated; a later non-idle status clears the transition so a new idle can ingest again.

Optional env overrides:

- `MEMORYMESH_URL` (default from install)
- `MEMORYMESH_PROJECT` (default from install)

## Provider-to-provider transfer

You can package one provider's memory context and send it to another:

```bash
memorymesh transfer --project deepiri --from claude --to opencode
memorymesh transfer --project deepiri --from claude --to cursor --push
memorymesh go --project deepiri --from claude --to gemini
```

- Without `--push`, it writes a transfer JSON bundle.
- With `--push`, it also calls `memorymesh-bridge-<target>` if installed (file-bridge targets).

## Supported integration targets

- `cursor`
- `claude`
- `gemini`
- `opencode` (native plugin)
- `continue`
