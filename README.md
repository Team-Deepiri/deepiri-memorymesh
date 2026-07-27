# Deepiri MemoryMesh

A local-first memory layer for AI agents, coding tools, and internal automation. Access it through Python, CLI, local HTTP, TUI, and provider integrations.

Memory Mesh is **not** a hosted multi-user SaaS product and **not** a production network server. The HTTP service binds to loopback only.

## Installation

```bash
pip install memorymesh
```

Runtime dependencies: **Typer** (CLI) and **PyYAML** (platform configuration).

Optional semantic embeddings:

```bash
pip install memorymesh[embeddings]
```

That installs `sentence-transformers`. Without it, the shared embedder falls back to a local hash backend (with a clear warning when `auto` / `sentence-transformers` was requested).

Optional encryption at rest:

```bash
pip install memorymesh[security]
```

That installs `cryptography` (AES-256-GCM). Plaintext mode works without it; enabling encryption without the extra fails clearly.

## Python quick start

```python
from memorymesh import Memory

mem = Memory()
mem.store("customer prefers Azure")
mem.query("Which cloud does the customer prefer?")
# → ["customer prefers Azure"]
mem.all()
```

`Memory` is a convenience facade over the **same canonical platform schema and database** used by CLI/HTTP/TUI. It does not start an HTTP service and does not require YAML configuration.

| Method | Behavior |
|--------|----------|
| `Memory(db_path=None, embedder="auto", project="default")` | Default DB is the canonical platform path. `embedder` is `auto` or `fallback`. |
| `store(content)` | Inserts into the platform schema and embeds immediately (exact-content dedupe). |
| `query(text, top_k=3)` | Returns content strings from the simple-API namespace. |
| `all()` | Lists facade-owned contents only. |
| `embedding_status()` | Requested vs active backend diagnostics. |

## Canonical storage

Default database:

```text
~/.config/deepiri-memorymesh/memorymesh.db
```

Python, CLI, HTTP, TUI, and integrations share this core when pointed at the same database and project. Facade rows use a stable namespace (`provider=python`, `conversation_id=python-api`, `role=memory`) so `Memory.all()` does not mix in arbitrary provider transcripts unless they share that namespace.

Optional platform config (created by `memorymesh init` / `Settings.load()`):

```text
~/.config/deepiri-memorymesh/config.yaml
```

The simple `Memory()` facade does **not** create or rewrite that YAML by itself.

## CLI quick start

```bash
memorymesh init
memorymesh ingest --provider cursor --project myapp --file ./export.json
memorymesh sync --provider cursor --project myapp --source-dir ~/.cursor
memorymesh embed --project myapp
memorymesh query --project myapp --q "auth decisions"
memorymesh query --project myapp --q "auth decisions" --mode indexed
memorymesh embedding-status
memorymesh database-status
memorymesh migrations status
memorymesh search-index status
memorymesh stats --project myapp
memorymesh serve --host 127.0.0.1 --port 8765
memorymesh tui --project myapp
memorymesh tui --project myapp --with-service
```

Useful related commands: `sync-auto`, `pipeline`, `compress`, `bundle export` / `bundle import`, `import-legacy-memory`, `install-native`, `uninstall-native`, `transfer`, `provider-health`, `migrations apply`, `search-index rebuild`, `encryption status`, `auth token create`, `pull api`.

## Schema migrations

The platform database uses an ordered, versioned migration history (`schema_migrations`). New databases migrate to the latest schema on `init`. Existing Batch 4 unversioned platform databases are adopted without rewriting user rows. Legacy/simple, mixed, corrupt, and future schemas fail closed.

```bash
memorymesh migrations status
memorymesh migrations apply --dry-run
memorymesh migrations apply
```

Nonempty upgrades create a consistent SQLite backup (WAL checkpoint + backup API) before changing schema.

## Query pipeline

Two paths matter:

1. **Simple `Memory.store()`** — writes a platform message **and** a versioned embedding immediately, so `query()` works without a separate embed step.
2. **CLI / provider ingest** — stores messages first. Platform semantic retrieval needs embeddings: run `memorymesh embed --project …` or `memorymesh pipeline --project …` (optionally with `--auto-sync`).

Retrieval modes (`exact` / `indexed` / `auto`):

- **exact** — score every compatible embedding in scope (full recall)
- **indexed** — lexical candidate index in SQLite, then strict vector rerank (bounded; not ANN)
- **auto** — exact below `retrieval_exact_threshold` (default 200), indexed above it

Candidate limits default to `retrieval_candidate_limit` (64). Query reports include strategy requested/used and candidate counts.

## Local HTTP security

`memorymesh serve`:

- binds **loopback only** (IPv4 and IPv6)
- rejects non-loopback hosts (no remote mode)
- restricts file ingest to repeatable `--ingest-root` allowlists (traversal/symlink escapes rejected)
- enforces a **10 MiB** request body limit
- requires **project-scoped HTTP bearer tokens** by default for project endpoints (`/ingest`, `/query`, `/stats`, `/state/*`)
- `/health` remains unauthenticated and never exposes the database path
- `--auth-mode off` is an explicit local trusted-user opt-out that prints a warning (never implied by a missing/malformed token)

```bash
memorymesh auth token create --project myapp --scope read --scope write --token-file ~/.config/deepiri-memorymesh/tokens/myapp.token
memorymesh serve --host 127.0.0.1 --port 8765
```

Direct Python/CLI/TUI database access stays trusted-local for the OS user and does not require HTTP tokens. Bridges read `MEMORYMESH_TOKEN` / `MEMORYMESH_TOKEN_FILE` at runtime (never embed token literals).

Inline conversation ingest (e.g. OpenCode plugin) remains supported without a filesystem path.

## Optional encryption at rest

```bash
memorymesh encryption key-generate --key-file ~/.config/deepiri-memorymesh/encryption.key
memorymesh encryption enable --key-file ~/.config/deepiri-memorymesh/encryption.key
memorymesh encryption status
```

Requires `pip install memorymesh[security]`. Protects stolen/copied DB files and backups without the key. Does **not** protect same-user malware, key theft, root, or full workstation compromise. Schema/ids/timestamps remain visible. Enable leaves a plaintext-sensitive pre-encryption backup in place. Losing the key loses decryptable content.

## Explicit API pull (generic)

```bash
memorymesh pull api --url https://example.com/export.jsonl \
  --project myapp --format jsonl --allow-host example.com \
  --token-env MY_PULL_TOKEN
```

Supports only generic `json` / `jsonl` / Memory Mesh `bundle` payloads — not provider account-history APIs. HTTPS-only by default; private/loopback needs `--allow-private`.

## Device scan & portable packaging

Scan Claude Code, Cursor, and OpenCode data across your machine (not just the repo):

```bash
# Discover locations
memorymesh scan

# Ingest all provider messages
memorymesh pull -p myproject

# Build portable package for another machine/provider
memorymesh package build -p myproject -o ./udata.tar.gz

# Import on another machine
memorymesh package import ./udata.tar.gz -p myproject
```

See [docs/U_DATA_PACKAGING.md](docs/U_DATA_PACKAGING.md) and [docs/STORAGE_PATHS.md](docs/STORAGE_PATHS.md).

## Export chat & memory

Export everything for a project (messages, summaries, agent state) as plain text, Markdown, or JSON:

```bash
memorymesh export -p myproject --format md -o ./export.md
memorymesh export -p myproject --format txt --clipboard
memorymesh export -p myproject --format json --provider cursor
```

The TUI (`memorymesh tui`) adds **[7] Export**. The HTTP API accepts `POST /export` with `{"project":"...", "format":"md", "clipboard": true}`.

## Cross-provider chat transfer

Move a conversation from one tool to another without dumping the whole project:

```bash
memorymesh init

# Workspace Session Bridge — auto-picks the latest session for this cwd
memorymesh resume -p lighthouse --from claude --to cursor -w ~/lighthouse

# Or target any provider pair dynamically
memorymesh resume -p myrepo --from claude --to gemini

# Manual conversation filter still supported
memorymesh transfer -p lighthouse --from claude --to cursor -c e7fcaea3 --push
```

**How it works (novel bits):**
- Correlates sessions by **workspace slug** (`~/.claude/projects/-home-user-repo/` ↔ workspace path)
- Builds a **resume brief** (compressed summary + tail turns) instead of 10k+ message dumps
- Delivers via a **dynamic provider registry** — handoff paths and import hints resolve per target, with a generic fallback for unknown agents

See [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) for `transfer`, `transfer-deliver`, and `install-push`.

## TUI

`memorymesh tui` runs the direct local curses UI and does **not** auto-start a detached HTTP server. An already-running compatible loopback service may be detected but is not owned or stopped. Use `memorymesh tui --with-service` for a supervised in-process service that always shuts down when the TUI exits. Long-running HTTP remains `memorymesh serve`.

## Integrations

Summaries and installers: [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md).

Provider parser matrix and maturity: [docs/PROVIDERS.md](docs/PROVIDERS.md).

Highlights:

- Transactional `install-native` / `uninstall-native` with manifests under `~/.config/deepiri-memorymesh/integrations/`
- OpenCode native plugin: `~/.config/opencode/plugins/memorymesh.ts`
- Bridges/hooks for Cursor, Claude, Gemini, Continue; Aider chat-history parser with stable re-ingestion
- Non-Claude providers never fall back to Claude history
- Unsupported placeholder providers are skipped by sync-auto (not reported as success)

## Bundle export / import

```bash
memorymesh bundle export --project myapp --out ./bundle.json
memorymesh bundle import --bundle ./bundle.json
memorymesh bundle import --bundle ./bundle.json --project other
```

Bundles round-trip **messages and summaries**. Re-import is idempotent for messages (dedupe) and uses summary upsert for `(project, conversation_id)`. The source bundle file is never modified.

## Legacy simple database migration

Older releases used a separate one-table DB at `~/.memorymesh/memory.db`. That path is **not** migrated automatically.

```bash
memorymesh import-legacy-memory --dry-run
memorymesh import-legacy-memory --source ~/.memorymesh/memory.db --project default
```

- Source database is never modified or deleted
- Content is re-embedded with the active canonical embedder
- Rows land in the same ownership namespace as `Memory(project=…)` (`provider=python`, `conversation_id=python-api`, `role=memory`) so `Memory.all()` / `Memory.query()` see them immediately
- Re-runs skip duplicates
- Pointing `Memory(db_path=…)` at a legacy file raises a clear error instead of rewriting it

## Embedding compatibility

New embeddings use a **versioned JSON envelope** (backend, model, dimensions, vector). Raw JSON float arrays are treated as legacy.

- Requested vs active backends are visible via `embedding_status` / `memorymesh embedding-status`
- Auto fallback to the hash backend warns once per instance; explicit `fallback` stays quiet
- Incompatible or malformed rows are skipped during ranking (diagnostics included)
- Re-embed after backend changes: `memorymesh embed --project <name>`
- Older package versions that only understand raw arrays may not read versioned payloads (upgrade or re-embed)

## Development and tests

```bash
python -m pip install .
python -m pip install '.[security]'   # optional; needed for encryption tests
python -m unittest discover -s tests -v
python -W error::ResourceWarning -m unittest discover -s tests -v
python -m compileall deepiri_memorymesh memorymesh
python -m build --wheel
```

OpenCode plugin runtime tests compile the generated TypeScript with a **pinned local esbuild** (see `package.json`). They do not silently download tooling via bare `npx`. If Node/esbuild is missing locally, those tests skip with an explicit reason; CI installs Node and esbuild and runs them. CI also runs a dedicated job that installs `memorymesh[security]` and exercises encryption tests.

## Project limitations

- Local-first trust model (your machine, your files)
- SQLite scaling limits; lexical candidate index + exact vector rerank (not ANN)
- Unsupported placeholder providers are skipped, not pretended to work
- No multi-user isolation or remote multi-tenant service
- The public PyPI name `memorymesh` is already occupied by an unrelated project; rename/publish remains a separate decision
- HTTP tokens protect local HTTP project endpoints only — not direct OS-user DB access
- Optional field encryption protects copied DB files without the key; it is not full-disk encryption
