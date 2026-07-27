"""Memory Mesh public package: simple Memory facade over the platform core."""

from __future__ import annotations

import warnings
from pathlib import Path

from deepiri_memorymesh.config import default_db_path
from deepiri_memorymesh.embedding_codec import (
    RankReport,
    REEMBED_HINT,
    query_embedding_meta,
)
from deepiri_memorymesh.embeddings import Embedder, EmbeddingStatus, SUPPORTED_BACKENDS
from deepiri_memorymesh.models import MemoryRecord, now_iso
from deepiri_memorymesh.namespace import (
    DEFAULT_PROJECT,
    MemoryOwnership,
    simple_api_ownership,
)
from deepiri_memorymesh.retrieval import rank_rows_with_report
from deepiri_memorymesh.storage import (
    AmbiguousSchemaError,
    LegacySchemaError,
    MemoryStore,
    detect_db_schema,
)


def _resolve_embedder(embedder: str) -> tuple[str, Embedder]:
    """Map Memory embedder names onto the shared Embedder.

    ``auto`` tries sentence-transformers (which may fall back internally).
    ``fallback`` stays explicit. Unsupported values raise ValueError.
    """
    requested = embedder.strip() if embedder else "auto"
    if requested == "auto":
        # Memory owns the user-facing warning so requested stays ``auto``.
        return requested, Embedder(
            "sentence-transformers", emit_fallback_warning=False
        )
    if requested == "fallback":
        return requested, Embedder("fallback")
    if requested in SUPPORTED_BACKENDS:
        # Allow explicit platform backend names for convenience.
        return requested, Embedder(requested)
    raise ValueError(
        f"Unsupported embedder: {embedder!r}. Expected 'auto' or 'fallback'."
    )


class Memory:
    """Convenient public facade over the canonical Memory Mesh platform store.

    Uses ``~/.config/deepiri-memorymesh/memorymesh.db`` by default (same schema
    as CLI/HTTP/TUI). Does not start an HTTP service and does not load or rewrite
    YAML configuration.

    Ownership: every store/query/all call is scoped to the simple-API namespace
    (provider/project/conversation_id/role). ``query`` searches **only**
    facade-owned memories — the same logical collection as ``all()`` — not
    arbitrary provider transcripts that happen to share the project name.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        embedder: str = "auto",
        project: str = DEFAULT_PROJECT,
    ):
        self.db_path = Path(db_path).expanduser() if db_path else default_db_path()
        self.ownership: MemoryOwnership = simple_api_ownership(project)
        self.project = self.ownership.project
        self.provider = self.ownership.provider
        self.conversation_id = self.ownership.conversation_id
        self.role = self.ownership.role
        self.last_query_report: RankReport | None = None

        kind = detect_db_schema(self.db_path)
        if kind == "legacy":
            raise LegacySchemaError(self.db_path)
        if kind in {"unknown", "corrupt"}:
            raise AmbiguousSchemaError(self.db_path, kind)

        # Key material: MEMORYMESH_ENCRYPTION_KEY env, or optional key-file path env.
        # Avoid Settings.load() here so constructing Memory never rewrites YAML.
        import os

        key_file_env = os.environ.get("MEMORYMESH_ENCRYPTION_KEY_FILE")
        key_file = Path(key_file_env).expanduser() if key_file_env else None
        self._store = MemoryStore(self.db_path, key_file=key_file)
        self._store.init()

        self.requested_embedder, self._embedder = _resolve_embedder(embedder)
        self.fallback_occurred = self._embedder.fallback_occurred
        self.fallback_reason = self._embedder.fallback_reason
        # Mirror Embedder active backend for compatibility with prior tests/docs.
        self.active_embedder = self._embedder.backend
        self.model_id = self._embedder.model_id
        self._fallback_warned = False
        if self.fallback_occurred and self.requested_embedder == "auto":
            self._emit_fallback_warning()

    def _emit_fallback_warning(self) -> None:
        if self._fallback_warned:
            return
        self._fallback_warned = True
        reason = self.fallback_reason or "unknown"
        warnings.warn(
            f"Embedding backend fallback: requested={self.requested_embedder!r} "
            f"active={self.active_embedder!r} reason={reason}; "
            "retrieval quality may differ.",
            UserWarning,
            # Memory.__init__ -> _emit_fallback_warning -> warn
            stacklevel=3,
        )

    @property
    def dimensions(self) -> int:
        return self._embedder.dimensions

    def embedding_status(self) -> dict[str, object]:
        """Programmatic embedding status for diagnostics/tests (T30)."""
        status: EmbeddingStatus = self._embedder.status()
        return {
            "requested_backend": self.requested_embedder,
            "active_backend": status.active_backend,
            "model": status.model,
            "dimensions": status.dimensions,
            "fallback_occurred": status.fallback_occurred,
            "fallback_reason": status.fallback_reason,
        }

    def store(self, content: str) -> None:
        """Store a memory with an immediate versioned embedding.

        Exact-content duplicates within the simple-API namespace are ignored.
        Lookup, message insert, and embedding write run under one
        ``BEGIN IMMEDIATE`` transaction so concurrent ``Memory`` instances
        cannot double-insert the same facade content.
        """
        rec = MemoryRecord(
            provider=self.ownership.provider,
            project=self.ownership.project,
            conversation_id=self.ownership.conversation_id,
            role=self.ownership.role,
            content=content,
            timestamp=now_iso(),
            metadata_json='{"source":"python-api"}',
        )
        # Embed outside the lock; only durable writes are serialized.
        embedding_json = self._embedder.dumps(self._embedder.embed(content))
        self._store.store_facade_memory_with_embedding(
            rec=rec,
            embedding_json=embedding_json,
        )

    def query(
        self,
        query: str,
        top_k: int = 3,
        *,
        strategy: str | None = None,
        candidate_limit: int | None = None,
    ) -> list[str]:
        """Query **facade-owned** memories by semantic similarity.

        Only rows matching the simple-API ownership namespace are candidates —
        the same collection returned by :meth:`all`. Platform/provider messages
        in the same project are not searched. Returns content strings.
        Incompatible stored embeddings are skipped (see
        :attr:`last_query_report` / :data:`REEMBED_HINT`).
        """
        from deepiri_memorymesh.config import (
            DEFAULT_RETRIEVAL_CANDIDATE_LIMIT,
            DEFAULT_RETRIEVAL_EXACT_THRESHOLD,
            DEFAULT_RETRIEVAL_MODE,
        )
        from deepiri_memorymesh.search_index import (
            select_candidate_message_ids,
            tokenize_for_index,
        )
        from deepiri_memorymesh.sync_service import QueryReport

        requested = (strategy or DEFAULT_RETRIEVAL_MODE).strip().lower()
        if requested not in {"exact", "indexed", "auto"}:
            raise ValueError(
                f"Unsupported retrieval strategy: {strategy!r}. "
                "Expected 'exact', 'indexed', or 'auto'."
            )
        limit = (
            int(candidate_limit)
            if candidate_limit is not None
            else DEFAULT_RETRIEVAL_CANDIDATE_LIMIT
        )
        query_vec = self._embedder.embed(query)
        query_meta = query_embedding_meta(
            query_vec,
            backend=self._embedder.backend,
            model=self._embedder.model_id,
        )
        total_eligible = self._store.count_embeddings(
            self.ownership.project,
            provider=self.ownership.provider,
            conversation_id=self.ownership.conversation_id,
            role=self.ownership.role,
        )
        used = requested
        fallback_reason = None
        if requested == "auto":
            used = "exact" if total_eligible <= DEFAULT_RETRIEVAL_EXACT_THRESHOLD else "indexed"

        rows: list[dict]
        candidate_count = 0
        if used == "indexed":
            terms = self._store.map_query_terms(tokenize_for_index(query))
            if not terms:
                if requested == "indexed":
                    qreport = QueryReport(
                        strategy_requested=requested,
                        strategy_used="indexed",
                        total_eligible_embeddings=total_eligible,
                        candidate_limit=limit,
                        exact_fallback_reason="no_lexical_candidates; use exact mode for full recall",
                    )
                    self.last_query_report = qreport.rank
                    return []
                used = "exact"
                fallback_reason = "empty_query_tokens"
            else:
                with self._store.connection() as conn:
                    ids = select_candidate_message_ids(
                        conn,
                        terms=terms,
                        limit=limit,
                        project=self.ownership.project,
                        provider=self.ownership.provider,
                        conversation_id=self.ownership.conversation_id,
                        role=self.ownership.role,
                    )
                if not ids:
                    if requested == "indexed":
                        qreport = QueryReport(
                            strategy_requested=requested,
                            strategy_used="indexed",
                            total_eligible_embeddings=total_eligible,
                            candidate_limit=limit,
                            exact_fallback_reason="no_lexical_candidates; use exact mode for full recall",
                        )
                        self.last_query_report = qreport.rank
                        return []
                    used = "exact"
                    fallback_reason = "no_lexical_candidates"
                else:
                    candidate_count = len(ids)
                    rows = [
                        dict(r)
                        for r in self._store.list_embeddings_by_ids(
                            ids,
                            project=self.ownership.project,
                            provider=self.ownership.provider,
                            conversation_id=self.ownership.conversation_id,
                            role=self.ownership.role,
                        )
                    ]
        if used == "exact":
            rows = [
                dict(r)
                for r in self._store.list_embeddings_for_namespace(
                    project=self.ownership.project,
                    provider=self.ownership.provider,
                    conversation_id=self.ownership.conversation_id,
                    role=self.ownership.role,
                )
            ]
            candidate_count = len(rows)

        ranked, report = rank_rows_with_report(
            query_vec, rows, top_k=top_k, query_meta=query_meta
        )
        # Compatibility: expose RankReport-shaped diagnostics on the facade.
        self.last_query_report = report
        return [str(item["content"]) for item in ranked]


    def all(self) -> list[str]:
        """List facade-owned memory contents (simple-API namespace only)."""
        rows = self._store.list_messages_for_namespace(
            project=self.ownership.project,
            provider=self.ownership.provider,
            conversation_id=self.ownership.conversation_id,
            role=self.ownership.role,
        )
        return [str(r["content"]) for r in rows]


# Re-export canonical path helper for callers/tests.
from deepiri_memorymesh.config import DEFAULT_DB_PATH  # noqa: E402

__all__ = [
    "Memory",
    "REEMBED_HINT",
    "DEFAULT_DB_PATH",
    "LegacySchemaError",
    "AmbiguousSchemaError",
]
__version__ = "0.2.0"
