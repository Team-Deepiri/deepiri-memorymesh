"""Canonical embedding serialization, parsing, and strict similarity (T15).

Versioned payload format (new writes)::

    {
      "version": 1,
      "backend": "hash-v1" | "sentence-transformers/all-MiniLM-L6-v2" | ...,
      "model": "optional-model-id-or-null",
      "dimensions": 128,
      "vector": [0.1, 0.2, ...]
    }

Legacy compatibility policy:
- Raw JSON arrays are the only legacy format (backend unknown).
- Equal dimensions vs the query vector: allowed with ``legacy_compatible=True``
  and counted in query diagnostics (never pretend the backend is known).
- Dimension mismatch, known-backend mismatch, malformed payloads, or
  non-numeric vectors: rejected / skipped — never scored via truncating zip.
"""

from __future__ import annotations

import json
import math
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

EMBEDDING_PAYLOAD_VERSION = 1

BACKEND_HASH_V1 = "hash-v1"
BACKEND_ST_MINILM = "sentence-transformers/all-MiniLM-L6-v2"

HASH_V1_DIMENSIONS = 128
ST_MINILM_DIMENSIONS = 384

# Config / Embedder API names → stable stored identifiers.
CONFIG_BACKEND_TO_STABLE = {
    "fallback": BACKEND_HASH_V1,
    "hash-v1": BACKEND_HASH_V1,
    "sentence-transformers": BACKEND_ST_MINILM,
    BACKEND_ST_MINILM: BACKEND_ST_MINILM,
}

STABLE_BACKEND_DIMENSIONS = {
    BACKEND_HASH_V1: HASH_V1_DIMENSIONS,
    BACKEND_ST_MINILM: ST_MINILM_DIMENSIONS,
}

# Canonical model id for each stable backend (None / matching id accepted).
STABLE_BACKEND_MODELS = {
    BACKEND_HASH_V1: BACKEND_HASH_V1,
    BACKEND_ST_MINILM: BACKEND_ST_MINILM,
}


class EmbeddingCodecError(ValueError):
    """Malformed or invalid embedding payload."""


class EmbeddingIncompatibilityError(ValueError):
    """Vectors cannot be compared (dimension / backend / malformed)."""


@dataclass(slots=True, frozen=True)
class ParsedEmbedding:
    vector: list[float]
    version: int | None
    backend: str | None
    model: str | None
    dimensions: int
    legacy: bool


@dataclass(slots=True)
class RankReport:
    """Diagnostics from ranking stored embeddings against a query vector.

    Counts are kept separate so callers can distinguish healthy versioned hits,
    legacy unknown-backend compatibility scoring, and skip reasons.
    """

    scored_versioned: int = 0
    legacy_compatible: int = 0
    skipped_incompatible: int = 0
    skipped_malformed: int = 0
    reasons: list[str] | None = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []

    @property
    def scored(self) -> int:
        """Total embeddings that contributed a similarity score."""
        return self.scored_versioned + self.legacy_compatible

    @property
    def skipped(self) -> int:
        return self.skipped_incompatible + self.skipped_malformed


def stable_backend_id(config_or_stable: str) -> str:
    if not isinstance(config_or_stable, str) or not config_or_stable.strip():
        raise EmbeddingCodecError(f"Unsupported embedding backend: {config_or_stable!r}")
    key = config_or_stable.strip()
    if key in CONFIG_BACKEND_TO_STABLE:
        return CONFIG_BACKEND_TO_STABLE[key]
    raise EmbeddingCodecError(f"Unsupported embedding backend: {config_or_stable!r}")


def expected_dimensions(stable_backend: str) -> int | None:
    return STABLE_BACKEND_DIMENSIONS.get(stable_backend)


def _require_real_int(value: Any, field: str) -> int:
    if type(value) is not int:  # noqa: E721 — reject bool/float subclasses
        raise EmbeddingCodecError(f"embedding '{field}' must be an integer")
    return value


def _coerce_finite_floats(values: list[Any], *, context: str) -> list[float]:
    out: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise EmbeddingCodecError(f"{context} values must be finite numbers")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EmbeddingCodecError(f"{context} values must be finite numbers") from exc
        if not math.isfinite(number):
            raise EmbeddingCodecError(f"{context} values must be finite numbers")
        out.append(number)
    return out


def _canonicalize_model(stable: str, model: str | None) -> str:
    expected = STABLE_BACKEND_MODELS[stable]
    if model is None or model == "" or model == expected:
        return expected
    if model in CONFIG_BACKEND_TO_STABLE:
        resolved = CONFIG_BACKEND_TO_STABLE[model]
        if resolved == stable:
            return expected
    raise EmbeddingCodecError(
        f"embedding model {model!r} contradicts backend {stable!r}"
    )


def serialize_embedding(
    vector: Iterable[float],
    *,
    backend: str,
    model: str | None = None,
) -> str:
    """Serialize a vector as a strict versioned JSON envelope."""
    try:
        stable = stable_backend_id(backend)
    except EmbeddingCodecError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise EmbeddingCodecError(f"unsupported embedding backend: {backend!r}") from exc

    expected = STABLE_BACKEND_DIMENSIONS[stable]
    try:
        raw_list = list(vector)
    except TypeError as exc:
        raise EmbeddingCodecError("embedding vector must be iterable") from exc
    if not raw_list:
        raise EmbeddingCodecError("embedding vector must be non-empty")
    vec = _coerce_finite_floats(raw_list, context="embedding vector")
    if len(vec) != expected:
        raise EmbeddingCodecError(
            f"embedding vector length {len(vec)} != expected {expected} for {stable}"
        )
    model_id = _canonicalize_model(stable, model)
    payload = {
        "version": EMBEDDING_PAYLOAD_VERSION,
        "backend": stable,
        "model": model_id,
        "dimensions": expected,
        "vector": vec,
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def parse_embedding(raw: str | bytes | Any) -> ParsedEmbedding:
    """Parse a versioned envelope or legacy raw array.

    Only raw JSON arrays are treated as legacy. Versioned objects must include
    version, backend, dimensions, and vector with strict type checks. All
    failures are raised as :class:`EmbeddingCodecError`.
    """
    try:
        return _parse_embedding_inner(raw)
    except EmbeddingCodecError:
        raise
    except Exception as exc:
        raise EmbeddingCodecError(f"invalid embedding payload: {type(exc).__name__}") from exc


def _parse_embedding_inner(raw: str | bytes | Any) -> ParsedEmbedding:
    if isinstance(raw, (bytes, bytearray)):
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EmbeddingCodecError("embedding payload is not valid UTF-8") from exc
    elif isinstance(raw, str):
        text = raw
    else:
        raise EmbeddingCodecError("embedding payload must be a JSON string")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EmbeddingCodecError(f"invalid embedding JSON: {exc.msg}") from exc

    if isinstance(data, list):
        return _parse_legacy_array(data)

    if not isinstance(data, dict):
        raise EmbeddingCodecError("embedding payload must be an object or array")

    for required in ("version", "backend", "dimensions", "vector"):
        if required not in data:
            raise EmbeddingCodecError(f"embedding payload missing '{required}'")

    version_i = _require_real_int(data["version"], "version")
    if version_i != EMBEDDING_PAYLOAD_VERSION:
        raise EmbeddingCodecError(
            f"unsupported embedding version: {version_i} "
            f"(expected {EMBEDDING_PAYLOAD_VERSION})"
        )

    backend = data["backend"]
    if not isinstance(backend, str) or not backend.strip():
        raise EmbeddingCodecError("embedding 'backend' must be a non-empty string")
    if backend not in STABLE_BACKEND_DIMENSIONS:
        raise EmbeddingCodecError(f"unknown embedding backend: {backend!r}")
    stable = backend

    declared = _require_real_int(data["dimensions"], "dimensions")
    expected = STABLE_BACKEND_DIMENSIONS[stable]
    if declared != expected:
        raise EmbeddingCodecError(
            f"declared dimensions {declared} != expected {expected} for {stable}"
        )

    vector_raw = data["vector"]
    if not isinstance(vector_raw, list) or not vector_raw:
        raise EmbeddingCodecError("embedding 'vector' must be a non-empty array")
    vector = _coerce_finite_floats(vector_raw, context="embedding vector")
    if len(vector) != declared:
        raise EmbeddingCodecError(
            f"declared dimensions {declared} != vector length {len(vector)}"
        )
    if len(vector) != expected:
        raise EmbeddingCodecError(
            f"vector length {len(vector)} != expected {expected} for {stable}"
        )

    model = data.get("model")
    if model is not None and not isinstance(model, str):
        raise EmbeddingCodecError("embedding 'model' must be a string or null")
    model_s = _canonicalize_model(stable, model)

    return ParsedEmbedding(
        vector=vector,
        version=version_i,
        backend=stable,
        model=model_s,
        dimensions=len(vector),
        legacy=False,
    )


def _parse_legacy_array(data: list[Any]) -> ParsedEmbedding:
    if not data:
        raise EmbeddingCodecError("legacy embedding array must be non-empty")
    vector = _coerce_finite_floats(data, context="legacy embedding")
    return ParsedEmbedding(
        vector=vector,
        version=None,
        backend=None,
        model=None,
        dimensions=len(vector),
        legacy=True,
    )


def cosine_strict(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity requiring equal nonzero dimensions (no silent truncation)."""
    aa = list(a)
    bb = list(b)
    if not aa or not bb:
        raise EmbeddingIncompatibilityError("cosine requires non-empty vectors")
    if len(aa) != len(bb):
        raise EmbeddingIncompatibilityError(
            f"cosine dimension mismatch: {len(aa)} vs {len(bb)}"
        )
    try:
        aa_f = _coerce_finite_floats(list(aa), context="cosine")
        bb_f = _coerce_finite_floats(list(bb), context="cosine")
    except EmbeddingCodecError as exc:
        raise EmbeddingIncompatibilityError(str(exc)) from exc
    dot = sum(x * y for x, y in zip(aa_f, bb_f, strict=True))
    na = math.sqrt(sum(x * x for x in aa_f))
    nb = math.sqrt(sum(y * y for y in bb_f))
    if na == 0.0 or nb == 0.0:
        raise EmbeddingIncompatibilityError("cosine requires non-zero vectors")
    return dot / (na * nb)


def embeddings_compatible(
    query: ParsedEmbedding,
    stored: ParsedEmbedding,
) -> tuple[bool, str | None]:
    """Return whether *stored* may be scored against *query*, plus a skip reason.

    When the query backend is unknown, only legacy (unknown-backend) stored
    embeddings may be scored. Versioned stored rows are skipped so they are
    never counted as healthy ``scored_versioned`` hits.
    """
    if query.dimensions != stored.dimensions:
        return False, (
            f"dimension mismatch: query={query.dimensions} stored={stored.dimensions}"
        )
    if stored.legacy or stored.backend is None:
        return True, "legacy_same_dimension"
    if query.backend is None:
        return False, "query metadata missing; cannot score versioned embedding"
    if stored.backend != query.backend:
        return False, (
            f"backend mismatch: query={query.backend!r} stored={stored.backend!r}"
        )
    return True, None


def query_embedding_meta(
    vector: list[float],
    *,
    backend: str,
    model: str | None = None,
) -> ParsedEmbedding:
    stable = stable_backend_id(backend)
    model_id = _canonicalize_model(stable, model)
    expected = STABLE_BACKEND_DIMENSIONS[stable]
    if len(vector) != expected:
        raise EmbeddingCodecError(
            f"query vector length {len(vector)} != expected {expected} for {stable}"
        )
    return ParsedEmbedding(
        vector=list(vector),
        version=EMBEDDING_PAYLOAD_VERSION,
        backend=stable,
        model=model_id,
        dimensions=len(vector),
        legacy=False,
    )


_legacy_compat_warned = False
_legacy_compat_lock = threading.Lock()


def warn_legacy_compatible_once() -> None:
    """Emit at most one process-wide legacy-compatibility warning (thread-safe)."""
    global _legacy_compat_warned
    with _legacy_compat_lock:
        if _legacy_compat_warned:
            return
        _legacy_compat_warned = True
    warnings.warn(
        "Scored legacy embedding(s) with matching dimensions but unknown backend; "
        "re-embed the project with the active backend for consistent retrieval.",
        UserWarning,
        stacklevel=2,
    )


def sanitize_bridge_diagnostic(text: str, *, max_len: int = 160) -> str:
    """Bound and sanitize bridge stderr for public reports/logs.

    Uses the last non-empty sanitized line so multiline dumps (transcripts,
    payloads) are not concatenated into the public diagnostic.
    """
    lines: list[str] = []
    for line in (text or "").splitlines() or [text or ""]:
        cleaned_chars: list[str] = []
        for ch in line:
            if ch.isprintable() and ch not in "\r\n\t":
                cleaned_chars.append(ch)
            elif ch.isspace():
                cleaned_chars.append(" ")
        cleaned = " ".join("".join(cleaned_chars).split())
        if cleaned:
            lines.append(cleaned)
    detail = lines[-1] if lines else ""
    if len(detail) > max_len:
        return detail[:max_len] + "..."
    return detail


REEMBED_HINT = (
    "Re-embed the project with the currently active backend "
    "(e.g. `memorymesh embed --project <name>`) to replace incompatible payloads."
)
