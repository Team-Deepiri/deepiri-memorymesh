from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Iterable

from .embedding_codec import (
    BACKEND_HASH_V1,
    BACKEND_ST_MINILM,
    HASH_V1_DIMENSIONS,
    ST_MINILM_DIMENSIONS,
    serialize_embedding,
    stable_backend_id,
)


SUPPORTED_BACKENDS = frozenset({"fallback", "sentence-transformers"})


def _hash_embedding(text: str, dims: int = HASH_V1_DIMENSIONS) -> list[float]:
    vec = [0.0] * dims
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = digest[0] % dims
        sign = 1.0 if (digest[1] % 2 == 0) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@dataclass(slots=True, frozen=True)
class EmbeddingStatus:
    """Programmatic embedding backend status (T30)."""

    requested_backend: str
    active_backend: str
    model: str | None
    dimensions: int | None
    fallback_occurred: bool
    fallback_reason: str | None

    @property
    def stable_backend_id(self) -> str:
        return stable_backend_id(self.active_backend)


class Embedder:
    def __init__(
        self,
        backend: str = "fallback",
        *,
        emit_fallback_warning: bool = True,
    ):
        requested = backend.strip() if backend else "fallback"
        if requested not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported embedding backend: {backend!r}. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_BACKENDS))}."
            )

        self.requested_backend = requested
        self.backend = requested
        self.model = None
        self.model_id: str | None = None
        self.fallback_occurred = False
        self.fallback_reason: str | None = None
        self._fallback_warned = False
        self._emit_fallback_warning_enabled = emit_fallback_warning

        if requested == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]

                model_name = BACKEND_ST_MINILM
                self.model = SentenceTransformer(model_name)
                self.model_id = model_name
            except Exception as exc:
                self.backend = "fallback"
                self.model = None
                self.model_id = None
                self.fallback_occurred = True
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                self._emit_fallback_warning()
        else:
            # Explicit fallback: no failure warning.
            self.model_id = BACKEND_HASH_V1

    def _emit_fallback_warning(self) -> None:
        if self._fallback_warned or not self._emit_fallback_warning_enabled:
            return
        self._fallback_warned = True
        reason = self.fallback_reason or "unknown"
        warnings.warn(
            f"Embedding backend fallback: requested={self.requested_backend!r} "
            f"active={'fallback'!r} reason={reason}; retrieval quality may differ.",
            UserWarning,
            # __init__ -> _emit_fallback_warning -> warn; point at Embedder(...) caller.
            stacklevel=3,
        )

    @property
    def dimensions(self) -> int:
        if self.backend == "sentence-transformers":
            return ST_MINILM_DIMENSIONS
        return HASH_V1_DIMENSIONS

    @property
    def stable_backend_id(self) -> str:
        return stable_backend_id(self.backend)

    def status(self) -> EmbeddingStatus:
        return EmbeddingStatus(
            requested_backend=self.requested_backend,
            active_backend=self.backend,
            model=self.model_id,
            dimensions=self.dimensions,
            fallback_occurred=self.fallback_occurred,
            fallback_reason=self.fallback_reason,
        )

    def embed(self, text: str) -> list[float]:
        if self.backend == "sentence-transformers" and self.model is not None:
            arr = self.model.encode([text], normalize_embeddings=True)[0]
            return [float(x) for x in arr.tolist()]
        return _hash_embedding(text)

    def dumps(self, vector: Iterable[float]) -> str:
        """Serialize using the versioned embedding envelope for the active backend."""
        return serialize_embedding(
            vector,
            backend=self.backend,
            model=self.model_id,
        )
