"""Dead simple memory for AI agents."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

DEFAULT_DB_PATH = Path.home() / ".memorymesh" / "memory.db"


class Memory:
    """Drop-in memory layer for ANY AI agent."""

    def __init__(self, db_path: str | Path | None = None, embedder: str = "auto"):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db(embedder)

    def _init_db(self, embedder: str) -> None:
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL UNIQUE,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()
        conn.close()

        if embedder == "auto":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self._embed = self._embed_st
            except Exception:
                self._model = None
                self._embed = self._embed_fallback

    def _embed_st(self, text: str) -> list[float]:
        import json

        arr = self._model.encode([text], normalize_embeddings=True)[0]
        return json.loads(json.dumps(arr.tolist()))

    def _embed_fallback(self, text: str) -> list[float]:
        import hashlib
        import math

        dims = 128
        vec = [0.0] * dims
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = digest[0] % dims
            sign = 1.0 if (digest[1] % 2 == 0) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def store(self, content: str) -> None:
        """Store a memory."""
        import json
        import sqlite3

        embedding = self._embed(content)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR IGNORE INTO memories (content, embedding, created_at) VALUES (?, ?, ?)",
            (content, json.dumps(embedding), str(uuid.uuid4())),
        )
        conn.commit()
        conn.close()

    def query(self, query: str, top_k: int = 3) -> list[str]:
        """Query memories by semantic similarity."""
        import json
        import math
        import sqlite3

        query_vec = self._embed(query)

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        conn = sqlite3.connect(self.db_path)
        cur = conn.execute("SELECT content, embedding FROM memories")
        rows = list(cur.fetchall())
        conn.close()

        scored = []
        for content, emb_json in rows:
            emb = json.loads(emb_json)
            score = cosine(query_vec, emb)
            scored.append((score, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in scored[:top_k]]

    def all(self) -> list[str]:
        """List all stored memories."""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cur = conn.execute("SELECT content FROM memories")
        rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows


__all__ = ["Memory"]
__version__ = "0.1.0"