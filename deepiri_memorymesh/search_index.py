"""Portable lexical candidate index for bounded retrieval (T32).

Uses ordinary SQLite tables (not FTS5) for Python 3.10–3.13 portability.
Tokenization is shared by migration backfill, inserts, and query candidate
selection.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

# Minimum token length after casefold. Shorter tokens are dropped.
MIN_TERM_LENGTH = 2
# Hard cap on characters per term (Unicode code points via len after casefold).
MAX_TERM_LENGTH = 64
# Hard cap on distinct terms stored per message.
MAX_TERMS_PER_MESSAGE = 256

# Small English function-word set. Documented policy: casefolded tokens in this
# set are excluded from the index. Non-English text is indexed fully aside from
# length bounds. This is intentionally not locale-dependent stemming.
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "of",
        "as",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
    }
)

# Letters/numbers across Unicode; apostrophes inside tokens kept.
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def tokenize_for_index(text: str) -> list[str]:
    """Deterministic, Unicode-safe tokenizer for the candidate index.

    Policy:
    - casefold (not locale-dependent lower)
    - extract alphanumeric-ish tokens via Unicode word characters
    - drop empty / stop-word / too-short / too-long tokens
    - preserve first-seen order; dedupe
    - bound distinct terms per message
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        raw = match.group(0).casefold()
        if len(raw) < MIN_TERM_LENGTH or len(raw) > MAX_TERM_LENGTH:
            continue
        if raw in STOP_WORDS:
            continue
        if raw in seen:
            continue
        seen.add(raw)
        out.append(raw)
        if len(out) >= MAX_TERMS_PER_MESSAGE:
            break
    return out


def index_message_terms(
    conn: sqlite3.Connection,
    message_id: int,
    content: str,
    *,
    term_mapper: Callable[[str], str] | None = None,
) -> int:
    """Replace term rows for *message_id* with tokens from *content*.

    When encryption is enabled, *term_mapper* should turn each plaintext term
    into a keyed HMAC token so plaintext search terms never land in SQLite.
    Must be called inside the caller's transaction. Returns number of terms written.
    """
    conn.execute("DELETE FROM memory_message_terms WHERE message_id = ?", (message_id,))
    terms = tokenize_for_index(content)
    if not terms:
        return 0
    if term_mapper is not None:
        terms = [term_mapper(t) for t in terms]
    conn.executemany(
        "INSERT OR IGNORE INTO memory_message_terms (message_id, term) VALUES (?, ?)",
        [(message_id, term) for term in terms],
    )
    return len(terms)


def backfill_message_terms(conn: sqlite3.Connection) -> int:
    """Index all messages that lack term rows. Returns messages (re)indexed."""
    # Ensure table exists (migration creates it; repair paths may call this too).
    tables = {
        str(r[0])
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "memory_message_terms" not in tables:
        return 0
    rows = conn.execute(
        """
        SELECT m.id, m.content
        FROM memory_messages m
        WHERE NOT EXISTS (
            SELECT 1 FROM memory_message_terms t WHERE t.message_id = m.id
        )
        ORDER BY m.id ASC
        """
    ).fetchall()
    count = 0
    for message_id, content in rows:
        index_message_terms(conn, int(message_id), str(content or ""))
        count += 1
    return count


def rebuild_all_message_terms(conn: sqlite3.Connection) -> int:
    """Drop and rebuild the entire term index. Returns messages indexed."""
    conn.execute("DELETE FROM memory_message_terms")
    rows = conn.execute(
        "SELECT id, content FROM memory_messages ORDER BY id ASC"
    ).fetchall()
    count = 0
    for message_id, content in rows:
        index_message_terms(conn, int(message_id), str(content or ""))
        count += 1
    return count


@dataclass(slots=True)
class SearchIndexStatus:
    messages: int
    indexed_messages: int
    term_rows: int
    missing_messages: int
    # "plaintext" or "keyed" (HMAC tokens). Never exposes key material.
    terms_mode: str = "plaintext"

    @property
    def complete(self) -> bool:
        return self.missing_messages == 0


def search_index_status(conn: sqlite3.Connection) -> SearchIndexStatus:
    tables = {
        str(r[0])
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    terms_mode = "plaintext"
    if "encryption_meta" in tables:
        row = conn.execute(
            "SELECT terms_mode FROM encryption_meta WHERE id = 1"
        ).fetchone()
        if row is not None and row[0]:
            terms_mode = str(row[0])
    if "memory_messages" not in tables:
        return SearchIndexStatus(0, 0, 0, 0, terms_mode=terms_mode)
    messages = int(conn.execute("SELECT COUNT(*) FROM memory_messages").fetchone()[0])
    if "memory_message_terms" not in tables:
        return SearchIndexStatus(messages, 0, 0, messages, terms_mode=terms_mode)
    term_rows = int(conn.execute("SELECT COUNT(*) FROM memory_message_terms").fetchone()[0])
    indexed = int(
        conn.execute(
            "SELECT COUNT(DISTINCT message_id) FROM memory_message_terms"
        ).fetchone()[0]
    )
    missing = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM memory_messages m
            WHERE NOT EXISTS (
                SELECT 1 FROM memory_message_terms t WHERE t.message_id = m.id
            )
            """
        ).fetchone()[0]
    )
    return SearchIndexStatus(
        messages=messages,
        indexed_messages=indexed,
        term_rows=term_rows,
        missing_messages=missing,
        terms_mode=terms_mode,
    )


def select_candidate_message_ids(
    conn: sqlite3.Connection,
    *,
    terms: list[str],
    limit: int,
    project: str,
    provider: str | None = None,
    conversation_id: str | None = None,
    role: str | None = None,
) -> list[int]:
    """Rank message ids by term overlap within the query scope.

    Only messages that already have an embedding row are candidates. The
    embedding join happens **before** LIMIT so unembedded lexical matches cannot
    exhaust the candidate budget. Deterministic order: overlap DESC, message_id
    ASC. Empty *terms* yields an empty candidate set.
    """
    if not terms or limit <= 0:
        return []
    placeholders = ",".join("?" for _ in terms)
    filters = ["m.project = ?"]
    params: list[object] = list(terms)
    params.append(project)
    if provider is not None:
        filters.append("m.provider = ?")
        params.append(provider)
    if conversation_id is not None:
        filters.append("m.conversation_id = ?")
        params.append(conversation_id)
    if role is not None:
        filters.append("m.role = ?")
        params.append(role)
    params.append(int(limit))
    where = " AND ".join(filters)
    sql = f"""
        SELECT t.message_id AS message_id, COUNT(DISTINCT t.term) AS overlap
        FROM memory_message_terms t
        JOIN memory_messages m ON m.id = t.message_id
        JOIN memory_embeddings e ON e.message_id = m.id
        WHERE t.term IN ({placeholders}) AND {where}
        GROUP BY t.message_id
        ORDER BY overlap DESC, t.message_id ASC
        LIMIT ?
    """
    rows = conn.execute(sql, params).fetchall()
    return [int(r[0]) for r in rows]
