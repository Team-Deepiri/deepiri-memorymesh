# deepiri-memorymesh

Cross-provider memory sync for coding assistants and agent tools.

`deepiri-memorymesh` lets you ingest conversation exports/context files from multiple tools,
store them as persistent memory, compress history, embed searchable chunks, and share agent
state across providers and local models.

It also supports non-CLI app integrations via a local service API + bridge scripts for
third-party extension/plugin hooks.

## What it supports

- Cross-provider context ingestion (initial adapters included)
- Cross-model memory syncing through one shared memory store
- Conversation compression pipelines (extractive + rolling summary)
- Persistent memory layers (raw messages, compressed summaries, agent state)
- Retrieval with embeddings and hybrid search
- Agent state sharing (project/session-level state snapshots)

## Providers and tools covered

Implemented or scaffolded adapters:

- Claude / Anthropic exports (dedicated parser)
- Gemini exports
- OpenAI ChatGPT exports
- Cursor chat/context exports (dedicated parser)
- OpenCode-style JSON logs
- Generic JSONL conversation logs

Researched targets to add next (config placeholders included):

- GitHub Copilot Chat
- Continue.dev
- Aider logs
- Cline / Roo Code
- Sourcegraph Cody
- Perplexity exports
- Replit Agent history
- Local models via Ollama
- Local models via LM Studio
- Local models via llama.cpp server

Provider research notes and integration status:

- `docs/PROVIDERS.md`
- `docs/INTEGRATIONS.md`

## Quick start

```bash
cd deepiri-memorymesh
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Initialize storage:

```bash
memorymesh init
```

Ingest conversation files:

```bash
memorymesh ingest --provider claude --project deepiri --file /path/to/claude_export.json
memorymesh ingest --provider cursor --project deepiri --file /path/to/cursor_chat.jsonl
```

Bulk sync a provider folder:

```bash
memorymesh sync --provider cursor --project deepiri --source-dir ~/.cursor
```

Auto-sync all configured providers:

```bash
memorymesh sync-auto --project deepiri
```

Run compression:

```bash
memorymesh compress --project deepiri
```

Build embeddings:

```bash
memorymesh embed --project deepiri
```

Run full pipeline:

```bash
memorymesh pipeline --project deepiri --auto-sync
```

Run local integration service (for app extensions/plugins):

```bash
memorymesh serve --host 127.0.0.1 --port 8765
```

List/install app integrations:

```bash
memorymesh integrations
memorymesh install-integration --target cursor --project deepiri
memorymesh install-integration --target opencode --project deepiri
memorymesh generate-hook-snippets --project deepiri --out-dir ./memorymesh-hooks
```

Search memory:

```bash
memorymesh query --project deepiri --q "where did we discuss retrieval strategy?"
```

Inspect memory layers:

```bash
memorymesh stats --project deepiri
```

Share agent state:

```bash
memorymesh state put --project deepiri --agent cursor --key current_task --value "memory sync pipeline"
memorymesh state get --project deepiri --agent claude --key current_task
```

Export/import portable context bundles:

```bash
memorymesh bundle export --project deepiri --out ./deepiri.bundle.json
memorymesh bundle import --bundle ./deepiri.bundle.json --project deepiri_clone
```

## Configuration

Default config lives at:

- `~/.config/deepiri-memorymesh/config.yaml`

You can set:

- Storage path (SQLite)
- Embedding backend (`sentence-transformers` or deterministic fallback)
- Provider adapter mapping
- Provider source paths for auto-sync
- Compression pipeline parameters

## Notes

- This first version is local-first and file-based.
- It does not call closed provider APIs directly by default.
- You can add API pullers later in `deepiri_memorymesh/providers/`.
