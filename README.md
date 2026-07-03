# Deepiri MemoryMesh 

**Drop-in Memory Layer Library for your AI Agent**

```python
from memorymesh import Memory

mem = Memory()

mem.store("user likes rust")
mem.query("what does the user like?")
# → ["user likes rust"]
```

## Install

```bash
pip install memorymesh
```

For semantic embeddings (recommended):

```bash
pip install memorymesh[embeddings]
```

## Usage

```python
from memorymesh import Memory

mem = Memory()                    # uses ~/.memorymesh/memory.db
mem.store("user prefers dark mode")
mem.store("working on async refactor")

# Query by semantic similarity
mem.query("theme preference")
# → ["user prefers dark mode"]

mem.query("async work", top_k=5)
# → ["working on async refactor", ...]
```

## API

| Method | Description |
|--------|-------------|
| `Memory(db_path=None, embedder="auto")` | Create memory. Set `embedder="fallback"` to skip embeddings model. |
| `mem.store(content)` | Store a memory (deduped). |
| `mem.query(query, top_k=3)` | Query by semantic similarity. |
| `mem.all()` | List all memories. |

## Philosophy

- **Dead simple** – 2 methods: `store()` and `query()`
- **Plug-and-play** – works out of the box, no config
- **Local-first** – SQLite + optional sentence-transformers
- **Zero deps** – falls back to deterministic hashing if no model

Works with any AI agent, tool, or assistant.

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

Move a full conversation from one tool to another:

```bash
memorymesh init
memorymesh go --project myrepo --from cursor --to claude
```

That syncs source exports, builds a transfer bundle, writes paste-ready `context.md` to the target inbox, and ingests the chat under the target provider in MemoryMesh.

See [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) for `transfer`, `transfer-deliver`, and `install-push`.
