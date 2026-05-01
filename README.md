# MemoryMesh 🧠

**Drop-in memory layer for ANY AI agent**

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