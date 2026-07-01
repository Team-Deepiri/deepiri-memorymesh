# Code App Integrations (Beyond CLI)

`deepiri-memorymesh` supports direct integration with code apps via:

- local HTTP service (`memorymesh serve`)
- per-app bridge scripts (`memorymesh-bridge-<target>`)
- extension/plugin hook templates

This allows syncing from tools even when they run through third-party extensions.

## 1) Start service

```bash
memorymesh serve --host 127.0.0.1 --port 8765
```

API endpoints:

- `GET /health`
- `GET /stats?project=<name>`
- `POST /ingest` (file path or raw conversation payload)
- `POST /query`
- `POST /state/put`
- `POST /state/get`

## 2) Install integration bridge

```bash
memorymesh install-integration --target cursor --project deepiri
memorymesh install-integration --target opencode --project deepiri
memorymesh install-native-all --project deepiri
```

This installs:

- executable bridge script in `~/.local/bin/`
- integration template in `~/.config/deepiri-memorymesh/integrations/`

Generate ready-to-paste hook snippets:

```bash
memorymesh generate-hook-snippets --project deepiri --out-dir ./memorymesh-hooks
```

Generated files include:

- `cursor.tasks.json`
- `opencode.hook.json`
- `continue.command.json`
- `claude.alias.sh`
- `gemini.alias.sh`

## 3) Hook your app/plugin

Use your app's extension/plugin post-export command or task hook to run:

```bash
memorymesh-bridge-cursor /path/to/export.json
memorymesh-bridge-opencode /path/to/export.jsonl
```

The bridge posts into local service API and syncs messages into persistent memory.

## Provider-to-provider transfer

Package one provider's chat context and deliver it to another tool's inbox:

```bash
# Full workflow: sync source exports, compress, bundle, deliver, clipboard
memorymesh go --project deepiri --from claude --to cursor

# Manual steps
memorymesh sync-auto --project deepiri
memorymesh transfer --project deepiri --from claude --to cursor --push
memorymesh transfer-render --bundle ~/.config/deepiri-memorymesh/transfers/deepiri.claude-to-cursor.json --to cursor
memorymesh transfer-deliver --bundle ~/.config/deepiri-memorymesh/transfers/deepiri.claude-to-cursor.json --to cursor --clipboard
```

Delivery writes paste-ready files under `~/.config/deepiri-memorymesh/inbox/<target>/`:

- `context.md` — paste into the target chat
- `import.json` — provider-shaped conversation JSON
- `IMPORT_INSTRUCTIONS.txt` — next steps per provider

Install per-target push scripts:

```bash
memorymesh install-push --target cursor
memorymesh-push-cursor ~/.config/deepiri-memorymesh/transfers/deepiri.claude-to-cursor.json
```

HTTP API:

- `POST /transfer` with `{project, from_provider, to_provider, deliver: true}`
- `POST /transfer/deliver` with `{bundle_path, to_provider}`

## Supported integration targets

- `cursor`
- `claude`
- `gemini`
- `opencode`
- `continue`

## OpenCode integration note

For OpenCode, use third-party plugin hooks or command callbacks to invoke:

- `memorymesh-bridge-opencode <path-to-conversation-export>`

This ensures OpenCode sessions are represented in the same shared memory layer as other tools.
