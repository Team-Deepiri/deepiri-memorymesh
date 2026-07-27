from __future__ import annotations

from typing import Any

from .embedding_codec import (
    EmbeddingCodecError,
    EmbeddingIncompatibilityError,
    ParsedEmbedding,
    RankReport,
    REEMBED_HINT,
    cosine_strict,
    embeddings_compatible,
    parse_embedding,
    warn_legacy_compatible_once,
)


def cosine(a, b) -> float:
    """Strict cosine similarity (equal nonzero dimensions required)."""
    return cosine_strict(a, b)


def rank_rows(
    query_vec: list[float],
    rows: list[dict],
    top_k: int = 8,
    *,
    query_meta: ParsedEmbedding | None = None,
    report: RankReport | None = None,
) -> list[dict]:
    """Rank rows by cosine similarity, skipping incompatible embeddings.

    Incompatible or malformed rows are not scored. Pass *report* (or read the
    returned list plus a fresh :class:`RankReport` via
    :func:`rank_rows_with_report`) to observe skipped counts.
    """
    ranked, _ = rank_rows_with_report(
        query_vec, rows, top_k=top_k, query_meta=query_meta, report=report
    )
    return ranked


def rank_rows_with_report(
    query_vec: list[float],
    rows: list[dict],
    top_k: int = 8,
    *,
    query_meta: ParsedEmbedding | None = None,
    report: RankReport | None = None,
) -> tuple[list[dict], RankReport]:
    diag = report if report is not None else RankReport()
    if query_meta is None:
        query_meta = ParsedEmbedding(
            vector=list(query_vec),
            version=None,
            backend=None,
            model=None,
            dimensions=len(query_vec),
            legacy=False,
        )

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        raw = row.get("embedding_json")
        try:
            stored = parse_embedding(raw)
        except EmbeddingCodecError as exc:
            diag.skipped_malformed += 1
            diag.reasons.append(f"malformed: {exc}")
            continue

        ok, reason = embeddings_compatible(query_meta, stored)
        if not ok:
            diag.skipped_incompatible += 1
            diag.reasons.append(reason or "incompatible")
            continue

        is_legacy = reason == "legacy_same_dimension"
        if is_legacy:
            warn_legacy_compatible_once()

        try:
            score = cosine_strict(query_meta.vector, stored.vector)
        except EmbeddingIncompatibilityError as exc:
            diag.skipped_incompatible += 1
            diag.reasons.append(str(exc))
            continue

        if is_legacy:
            diag.legacy_compatible += 1
        elif query_meta.backend is None:
            # Defensive: never treat unknown-query hits as healthy versioned.
            diag.legacy_compatible += 1
            diag.reasons.append("query_backend_unknown_degraded")
        else:
            diag.scored_versioned += 1
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    for score, row in scored[:top_k]:
        item = dict(row)
        item["score"] = score
        out.append(item)
    return out, diag


def format_rank_diagnostic(report: RankReport) -> str | None:
    """Human-readable per-call diagnostic; never reads shared mutable state."""
    if (
        report.skipped_incompatible == 0
        and report.skipped_malformed == 0
        and report.legacy_compatible == 0
    ):
        return None
    parts: list[str] = []
    if report.scored_versioned:
        parts.append(f"scored {report.scored_versioned} versioned embedding(s)")
    if report.legacy_compatible:
        parts.append(
            f"scored {report.legacy_compatible} legacy embedding(s) with unknown backend"
        )
    if report.skipped_incompatible:
        parts.append(
            f"skipped {report.skipped_incompatible} incompatible embedding(s)"
        )
    if report.skipped_malformed:
        parts.append(f"skipped {report.skipped_malformed} malformed embedding(s)")
    parts.append(REEMBED_HINT)
    return "; ".join(parts)
