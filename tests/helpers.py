"""Shared test helpers for Memory Mesh (temporary HOME, legacy DBs, etc.)."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence
from unittest import mock


@contextmanager
def temp_home(home: Path) -> Iterator[Path]:
    """Point ``HOME`` at *home* for the duration of the context."""
    home.mkdir(parents=True, exist_ok=True)
    with mock.patch.dict(os.environ, {"HOME": str(home)}, clear=False):
        yield home


def make_legacy_db(
    path: Path,
    rows: Sequence[tuple[object, object, object]],
) -> Path:
    """Create a historical simple-memory SQLite database.

    *rows* are ``(content, embedding, created_at)`` triples. ``content`` may be
    ``None`` to exercise malformed-row handling.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                embedding TEXT,
                created_at TEXT
            )
            """
        )
        for content, embedding, created_at in rows:
            conn.execute(
                "INSERT INTO memories (content, embedding, created_at) VALUES (?, ?, ?)",
                (content, embedding, created_at),
            )
        conn.commit()
    finally:
        conn.close()
    return path
