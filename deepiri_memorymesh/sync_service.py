from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import subprocess
from typing import Callable

from .compression import compress_conversation
from .config import Settings
from .embedding_codec import (
    RankReport,
    REEMBED_HINT,
    query_embedding_meta,
    sanitize_bridge_diagnostic,
)
from .embeddings import Embedder, EmbeddingStatus
from .file_scan import collect_provider_files
from .models import AgentState, CompressedRecord, MemoryRecord, now_iso
from .providers import parse_provider_file
from .providers.registry import SyncAutoProviderOutcome, get_provider
from .retrieval import format_rank_diagnostic, rank_rows_with_report
from .search_index import select_candidate_message_ids, tokenize_for_index
from .storage import MemoryStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SyncFileFailure:
    """One file that failed during directory sync."""

    path: Path
    error_type: str
    message: str


@dataclass(slots=True)
class SyncDirectoryReport:
    """Detailed sync counts.

    - attempted: files selected for ingest
    - processed: files ingested without raising
    - failed: files that raised during ingest
    - inserted: total messages inserted from successful files
    """

    attempted: int = 0
    processed: int = 0
    failed: int = 0
    inserted: int = 0
    failures: list[SyncFileFailure] = field(default_factory=list)


@dataclass(slots=True)
class TransferPushReport:
    """Outcome of an optional bridge push during transfer (T35)."""

    attempted: bool = False
    success: bool = False
    provider: str = ""
    bridge_path: Path | None = None
    returncode: int | None = None
    message: str = ""


@dataclass(slots=True)
class BundleImportReport:
    """Per-call bundle import counts (T18)."""

    messages_seen: int = 0
    messages_inserted: int = 0
    messages_duplicate: int = 0
    summaries_seen: int = 0
    summaries_inserted: int = 0
    summaries_updated: int = 0
    malformed_messages: int = 0
    malformed_summaries: int = 0
    message_errors: list[str] = field(default_factory=list)
    summary_errors: list[str] = field(default_factory=list)

    @property
    def messages_imported(self) -> int:
        """Compatibility alias for :attr:`messages_inserted`."""
        return self.messages_inserted

    @property
    def summaries_imported(self) -> int:
        """Compatibility alias: inserted + updated summary upserts."""
        return self.summaries_inserted + self.summaries_updated


@dataclass(slots=True)
class QueryReport:
    """Per-call retrieval report (T15 + T32). Embeds RankReport fields."""

    rank: RankReport = field(default_factory=RankReport)
    strategy_requested: str = "auto"
    strategy_used: str = "exact"
    total_eligible_embeddings: int = 0
    candidate_message_count: int = 0
    embeddings_scored: int = 0
    candidate_limit: int = 0
    exact_fallback_reason: str | None = None

    @property
    def scored_versioned(self) -> int:
        return self.rank.scored_versioned

    @property
    def legacy_compatible(self) -> int:
        return self.rank.legacy_compatible

    @property
    def skipped_incompatible(self) -> int:
        return self.rank.skipped_incompatible

    @property
    def skipped_malformed(self) -> int:
        return self.rank.skipped_malformed

    @property
    def skipped(self) -> int:
        return self.rank.skipped

    @property
    def scored(self) -> int:
        return self.rank.scored


@dataclass(slots=True)
class QueryResult:
    """Retrieval results plus compatibility diagnostics (T15/T32)."""

    rows: list[dict]
    report: QueryReport

    @property
    def diagnostic(self) -> str | None:
        return format_rank_diagnostic(self.report.rank)


@dataclass(slots=True)
class SyncAutoReport:
    """Per-call sync-auto outcome distinguishing provider classifications (T19)."""

    outcomes: list[SyncAutoProviderOutcome] = field(default_factory=list)
    total_processed: int = 0
    total_failed: int = 0
    total_inserted: int = 0
    skipped_unsupported: int = 0
    skipped_missing_path: int = 0


class MemoryMesh:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = MemoryStore(
            settings.db_path,
            key_file=settings.encryption_key_file,
        )
        self.embedder = Embedder(settings.embedding_backend)
        self.last_query_report: QueryReport | None = None
        self.last_transfer_push: TransferPushReport | None = None

    def init(self) -> None:
        self.store.init()

    def embedding_status(self) -> EmbeddingStatus:
        return self.embedder.status()

    def ingest_file(self, provider: str, project: str, file_path: Path) -> int:
        records = parse_provider_file(provider, project, file_path)
        return self.store.insert_messages(records)

    def sync_directory(
        self,
        provider: str,
        project: str,
        directory: Path,
        recursive: bool = True,
        include_globs: list[str] | None = None,
        *,
        error_sink: list[SyncFileFailure] | None = None,
        on_error: Callable[[SyncFileFailure], None] | None = None,
    ) -> tuple[int, int]:
        """Sync provider export files under *directory*.

        Returns ``(processed, inserted)`` for backward compatibility:
        - processed: successfully ingested files
        - inserted: messages inserted from those files

        Failures do not abort the sync. They are logged and optionally
        collected via *error_sink* / *on_error*. Use
        :meth:`sync_directory_report` for full count semantics.
        """
        report = self.sync_directory_report(
            provider=provider,
            project=project,
            directory=directory,
            recursive=recursive,
            include_globs=include_globs,
            error_sink=error_sink,
            on_error=on_error,
        )
        return report.processed, report.inserted

    def sync_directory_report(
        self,
        provider: str,
        project: str,
        directory: Path,
        recursive: bool = True,
        include_globs: list[str] | None = None,
        *,
        error_sink: list[SyncFileFailure] | None = None,
        on_error: Callable[[SyncFileFailure], None] | None = None,
    ) -> SyncDirectoryReport:
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        patterns = include_globs or ["*.json", "*.jsonl"]
        unique_files = collect_provider_files(
            directory, patterns, recursive=recursive
        )
        report = SyncDirectoryReport(attempted=len(unique_files))
        for path in unique_files:
            try:
                report.inserted += self.ingest_file(
                    provider=provider, project=project, file_path=path
                )
                report.processed += 1
            except Exception as exc:
                failure = SyncFileFailure(
                    path=path,
                    error_type=type(exc).__name__,
                    message=str(exc) or repr(exc),
                )
                report.failed += 1
                report.failures.append(failure)
                logger.warning(
                    "sync_directory skipped %s (%s): %s",
                    path,
                    failure.error_type,
                    failure.message,
                )
                if error_sink is not None:
                    error_sink.append(failure)
                if on_error is not None:
                    try:
                        on_error(failure)
                    except Exception as cb_exc:
                        logger.warning(
                            "sync_directory on_error callback failed for %s "
                            "(%s): %s; original failure was %s: %s",
                            path,
                            type(cb_exc).__name__,
                            str(cb_exc) or repr(cb_exc),
                            failure.error_type,
                            failure.message,
                        )
                continue
        return report

    def compress_project(self, project: str) -> int:
        rows = self.store.list_messages(project)
        grouped: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            grouped[str(row["conversation_id"])].append(f'{row["role"]}: {row["content"]}')
        count = 0
        for conv_id, messages in grouped.items():
            text = "\n".join(messages)
            summary = compress_conversation(
                text,
                target_chars=self.settings.compression_target_chars,
            )
            if not summary:
                continue
            self.store.upsert_summary(
                CompressedRecord(
                    project=project,
                    conversation_id=conv_id,
                    summary=summary,
                    method="extractive-frequency",
                )
            )
            count += 1
        return count

    def embed_project(self, project: str) -> int:
        rows = self.store.list_messages(project)
        count = 0
        for row in rows:
            vector = self.embedder.embed(str(row["content"]))
            self.store.save_embedding(int(row["id"]), self.embedder.dumps(vector))
            count += 1
        return count

    def query(
        self,
        project: str,
        text: str,
        top_k: int = 8,
        *,
        strategy: str | None = None,
        candidate_limit: int | None = None,
    ) -> list[dict]:
        """Semantic retrieval; incompatible rows are skipped.

        Prefer :meth:`query_with_report` in threaded contexts. The compatibility
        attribute :attr:`last_query_report` is updated but must not be used to
        attribute diagnostics across concurrent callers.
        """
        result = self.query_with_report(
            project=project,
            text=text,
            top_k=top_k,
            strategy=strategy,
            candidate_limit=candidate_limit,
        )
        return result.rows

    def query_with_report(
        self,
        project: str,
        text: str,
        top_k: int = 8,
        *,
        strategy: str | None = None,
        candidate_limit: int | None = None,
        provider: str | None = None,
        conversation_id: str | None = None,
        role: str | None = None,
    ) -> QueryResult:
        """Return ranked rows plus the per-call QueryReport for this exact call."""
        requested = (strategy or self.settings.retrieval_mode or "auto").strip().lower()
        if requested not in {"exact", "indexed", "auto"}:
            raise ValueError(
                f"Unsupported retrieval strategy: {strategy!r}. "
                "Expected 'exact', 'indexed', or 'auto'."
            )
        limit = (
            int(candidate_limit)
            if candidate_limit is not None
            else int(self.settings.retrieval_candidate_limit)
        )
        threshold = int(self.settings.retrieval_exact_threshold)

        qvec = self.embedder.embed(text)
        query_meta = query_embedding_meta(
            qvec,
            backend=self.embedder.backend,
            model=self.embedder.model_id,
        )

        total_eligible = self.store.count_embeddings(
            project,
            provider=provider,
            conversation_id=conversation_id,
            role=role,
        )
        qreport = QueryReport(
            strategy_requested=requested,
            candidate_limit=limit,
            total_eligible_embeddings=total_eligible,
        )

        used = requested
        fallback_reason: str | None = None
        candidate_ids: list[int] | None = None

        if requested == "auto":
            if total_eligible <= threshold:
                used = "exact"
            else:
                used = "indexed"

        if used == "indexed":
            terms = self.store.map_query_terms(tokenize_for_index(text))
            if not terms:
                if requested == "indexed":
                    qreport.strategy_used = "indexed"
                    qreport.exact_fallback_reason = (
                        "no_lexical_candidates; use exact mode for full recall"
                    )
                    qreport.candidate_message_count = 0
                    qreport.embeddings_scored = 0
                    self.last_query_report = qreport
                    return QueryResult(rows=[], report=qreport)
                used = "exact"
                fallback_reason = "empty_query_tokens"
            else:
                with self.store.connection() as conn:
                    candidate_ids = select_candidate_message_ids(
                        conn,
                        terms=terms,
                        limit=limit,
                        project=project,
                        provider=provider,
                        conversation_id=conversation_id,
                        role=role,
                    )
                if not candidate_ids:
                    if requested == "indexed":
                        qreport.strategy_used = "indexed"
                        qreport.exact_fallback_reason = (
                            "no_lexical_candidates; use exact mode for full recall"
                        )
                        qreport.candidate_message_count = 0
                        qreport.embeddings_scored = 0
                        self.last_query_report = qreport
                        return QueryResult(rows=[], report=qreport)
                    used = "exact"
                    fallback_reason = "no_lexical_candidates"

        qreport.strategy_used = used
        qreport.exact_fallback_reason = fallback_reason

        if used == "exact":
            if provider is None and conversation_id is None and role is None:
                rows = [dict(r) for r in self.store.list_embeddings(project)]
            else:
                # Scoped exact scan via candidate-id path with all ids would still
                # load everything; use list_embeddings_for_namespace when possible.
                if provider is not None and conversation_id is not None:
                    rows = [
                        dict(r)
                        for r in self.store.list_embeddings_for_namespace(
                            project=project,
                            provider=provider,
                            conversation_id=conversation_id,
                            role=role,
                        )
                    ]
                else:
                    rows = [dict(r) for r in self.store.list_embeddings(project)]
                    if provider is not None:
                        rows = [r for r in rows if r.get("provider") == provider]
                    if conversation_id is not None:
                        rows = [
                            r for r in rows if r.get("conversation_id") == conversation_id
                        ]
            qreport.candidate_message_count = len(rows)
        else:
            assert candidate_ids is not None
            qreport.candidate_message_count = len(candidate_ids)
            rows = [
                dict(r)
                for r in self.store.list_embeddings_by_ids(
                    candidate_ids,
                    project=project,
                    provider=provider,
                    conversation_id=conversation_id,
                    role=role,
                )
            ]

        ranked, rank = rank_rows_with_report(
            qvec, rows, top_k=top_k, query_meta=query_meta
        )
        qreport.rank = rank
        qreport.embeddings_scored = rank.scored + rank.skipped
        # Compatibility only — production threaded callers must use the returned report.
        self.last_query_report = qreport
        if rank.skipped or rank.legacy_compatible:
            logger.warning(
                "query project=%s strategy=%s versioned=%s legacy=%s incompatible=%s "
                "malformed=%s; %s",
                project,
                used,
                rank.scored_versioned,
                rank.legacy_compatible,
                rank.skipped_incompatible,
                rank.skipped_malformed,
                REEMBED_HINT,
            )
        return QueryResult(rows=ranked, report=qreport)

    def sync_auto_report(
        self,
        project: str,
        *,
        recursive: bool = True,
    ) -> SyncAutoReport:
        """Sync configured providers with honest capability classification (T19)."""
        report = SyncAutoReport()
        for provider in self.settings.providers:
            key = provider.strip().lower()
            cap = get_provider(key)
            raw = self.settings.provider_paths.get(provider) or self.settings.provider_paths.get(
                key, ""
            )
            if cap.parser_kind == "unsupported" or not cap.automatic_discovery:
                if cap.parser_kind == "generic-explicit":
                    # Generic formats require an explicit path+intent; skip quiet auto.
                    outcome = SyncAutoProviderOutcome(
                        provider=key,
                        classification="generic-explicit",
                        skipped=True,
                        reason="generic provider requires explicit sync/ingest",
                        path=str(raw or ""),
                    )
                else:
                    outcome = SyncAutoProviderOutcome(
                        provider=key,
                        classification="unsupported",
                        skipped=True,
                        reason="unsupported provider skipped by sync-auto",
                        path=str(raw or ""),
                    )
                    report.skipped_unsupported += 1
                report.outcomes.append(outcome)
                continue
            if not raw:
                report.outcomes.append(
                    SyncAutoProviderOutcome(
                        provider=key,
                        classification="native",
                        skipped=True,
                        reason="missing configured path",
                    )
                )
                report.skipped_missing_path += 1
                continue
            source = Path(raw).expanduser()
            if not source.exists() or not source.is_dir():
                report.outcomes.append(
                    SyncAutoProviderOutcome(
                        provider=key,
                        classification="native",
                        skipped=True,
                        reason="configured path missing or not a directory",
                        path=str(source),
                    )
                )
                report.skipped_missing_path += 1
                continue
            globs = self.settings.provider_globs.get(
                key, self.settings.provider_globs.get(provider, ["**/*.json", "**/*.jsonl"])
            )
            try:
                sync_report = self.sync_directory_report(
                    provider=provider,
                    project=project,
                    directory=source,
                    recursive=recursive,
                    include_globs=globs,
                )
            except Exception as exc:
                report.outcomes.append(
                    SyncAutoProviderOutcome(
                        provider=key,
                        classification="native",
                        skipped=False,
                        reason=f"parser_failure:{type(exc).__name__}",
                        path=str(source),
                        failed=1,
                    )
                )
                report.total_failed += 1
                continue
            report.outcomes.append(
                SyncAutoProviderOutcome(
                    provider=key,
                    classification="native",
                    skipped=False,
                    processed=sync_report.processed,
                    failed=sync_report.failed,
                    inserted=sync_report.inserted,
                    path=str(source),
                    reason="ok" if not sync_report.failed else "partial_failure",
                )
            )
            report.total_processed += sync_report.processed
            report.total_failed += sync_report.failed
            report.total_inserted += sync_report.inserted
        return report

    def export_bundle(
        self,
        project: str,
        output_path: Path,
        *,
        allow_plaintext: bool = False,
    ) -> Path:
        messages = [dict(r) for r in self.store.list_messages(project)]
        summaries = [dict(r) for r in self.store.list_summaries(project)]
        # Strip embedding_id/id helpers that aren't part of the public bundle shape.
        for msg in messages:
            msg.pop("content_fingerprint", None)
        for summary in summaries:
            # Keep conversation_id/summary/method/created_at; drop internal id.
            summary.pop("id", None)
        payload = {
            "project": project,
            "messages": messages,
            "summaries": summaries,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plaintext = json.dumps(payload, ensure_ascii=True, indent=2)

        ctx = self.store.encryption_context()
        if ctx is not None and not allow_plaintext:
            from .crypto import encrypt_portable

            envelope = encrypt_portable(
                ctx,
                plaintext,
                purpose="bundle",
                identity=project,
            )
            wrapped = {
                "memorymesh_bundle": 1,
                "encrypted": True,
                "kid": ctx.key_id,
                "aad_project": project,
                "payload": envelope,
            }
            output_path.write_text(
                json.dumps(wrapped, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            return output_path

        if ctx is not None and allow_plaintext:
            import warnings

            warnings.warn(
                "Exporting a PLAINTEXT bundle from an encrypted database. "
                "Treat this file as sensitive.",
                UserWarning,
                stacklevel=2,
            )
        output_path.write_text(plaintext, encoding="utf-8")
        return output_path

    def import_bundle(self, bundle_path: Path, project_override: str | None = None) -> int:
        """Import messages (and summaries) from a bundle.

        Returns the number of newly inserted messages for backward compatibility.
        Prefer :meth:`import_bundle_with_report` for summary counts and malformed
        entry diagnostics.
        """
        return self.import_bundle_with_report(
            bundle_path, project_override=project_override
        ).messages_inserted

    def import_bundle_with_report(
        self,
        bundle_path: Path,
        project_override: str | None = None,
    ) -> BundleImportReport:
        """Import bundle messages and summaries; return a per-call report.

        Malformed entries are skipped without aborting the rest of the import.
        Summaries use :meth:`MemoryStore.upsert_summary` (idempotent update-or-insert).
        The bundle file is never modified. ``project_override`` does not mutate
        the parsed JSON object — override is applied only to written records.
        Accepts plaintext bundles and encrypted Memory Mesh envelopes.
        """
        report = BundleImportReport()
        raw_text = bundle_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            raise ValueError("Bundle root must be a JSON object")

        if payload.get("encrypted") is True and payload.get("memorymesh_bundle") == 1:
            ctx = self.store.encryption_context()
            if ctx is None:
                raise ValueError(
                    "Encrypted bundle requires an encryption key "
                    "(MEMORYMESH_ENCRYPTION_KEY or encryption_key_file)"
                )
            from .crypto import decrypt_portable

            aad_project = str(payload.get("aad_project") or payload.get("project") or "default")
            inner = decrypt_portable(
                ctx,
                str(payload.get("payload") or ""),
                purpose="bundle",
                identity=aad_project,
            )
            payload = json.loads(inner)
            if not isinstance(payload, dict):
                raise ValueError("Decrypted bundle root must be a JSON object")

        # Snapshot original project field so callers/tests can prove non-mutation.
        original_project = payload.get("project")
        project = project_override or str(original_project or "default")
        messages = payload.get("messages")
        if messages is None:
            messages = []
        if not isinstance(messages, list):
            raise ValueError("Bundle 'messages' must be a list when present")

        summaries = payload.get("summaries")
        if summaries is None:
            summaries = []
        if not isinstance(summaries, list):
            raise ValueError("Bundle 'summaries' must be a list when present")

        flattened: list[MemoryRecord] = []
        for idx, msg in enumerate(messages):
            report.messages_seen += 1
            if not isinstance(msg, dict):
                report.malformed_messages += 1
                report.message_errors.append(f"messages[{idx}]: not an object")
                continue
            content = str(msg.get("content") or "")
            if not content:
                report.malformed_messages += 1
                report.message_errors.append(f"messages[{idx}]: empty content")
                continue
            try:
                flattened.append(
                    MemoryRecord(
                        provider=str(msg.get("provider") or "bundle"),
                        project=project,
                        conversation_id=str(msg.get("conversation_id") or "bundle"),
                        role=str(msg.get("role") or "unknown"),
                        content=content,
                        timestamp=str(msg.get("timestamp") or now_iso()),
                        metadata_json=str(msg.get("metadata_json") or "{}"),
                        source_key=(
                            str(msg["source_key"])
                            if msg.get("source_key") not in (None, "")
                            else None
                        ),
                    )
                )
            except Exception as exc:
                report.malformed_messages += 1
                report.message_errors.append(
                    f"messages[{idx}]: {type(exc).__name__}: {exc}"
                )

        if flattened:
            inserted = self.store.insert_messages(flattened)
            report.messages_inserted = int(inserted)
            report.messages_duplicate = len(flattened) - report.messages_inserted

        for idx, summary in enumerate(summaries):
            report.summaries_seen += 1
            if not isinstance(summary, dict):
                report.malformed_summaries += 1
                report.summary_errors.append(f"summaries[{idx}]: not an object")
                continue
            text = str(summary.get("summary") or "")
            conversation_id = str(summary.get("conversation_id") or "")
            method = str(summary.get("method") or "")
            if not text or not conversation_id or not method:
                report.malformed_summaries += 1
                report.summary_errors.append(
                    f"summaries[{idx}]: missing summary, conversation_id, or method"
                )
                continue
            created_at = str(summary.get("created_at") or now_iso())
            try:
                outcome = self.store.upsert_summary(
                    CompressedRecord(
                        project=project,
                        conversation_id=conversation_id,
                        summary=text,
                        method=method,
                        created_at=created_at,
                    )
                )
                if outcome == "inserted":
                    report.summaries_inserted += 1
                else:
                    report.summaries_updated += 1
            except Exception as exc:
                report.malformed_summaries += 1
                report.summary_errors.append(
                    f"summaries[{idx}]: {type(exc).__name__}: {exc}"
                )

        # Defensive: never write back; assert in-memory project field unchanged.
        if payload.get("project") != original_project:
            raise RuntimeError("bundle import mutated parsed payload project field")

        return report

    def put_state(self, project: str, agent: str, key: str, value: str) -> None:
        self.store.set_agent_state(
            AgentState(project=project, agent=agent, key=key, value=value)
        )

    def get_state(self, project: str, agent: str, key: str) -> str | None:
        return self.store.get_agent_state(project, agent, key)

    def stats(self, project: str) -> dict[str, int]:
        return self.store.project_stats(project)

    def transfer(
        self,
        project: str,
        from_provider: str,
        to_provider: str,
        out_path: Path | None = None,
        push_via_bridge: bool = False,
    ) -> tuple[Path, int]:
        """Transfer context; returns ``(path, count)`` for compatibility.

        Prefer :meth:`transfer_with_report` so push diagnostics are tied to the
        same call. :attr:`last_transfer_push` is updated for compatibility only.
        """
        path, count, _push = self.transfer_with_report(
            project=project,
            from_provider=from_provider,
            to_provider=to_provider,
            out_path=out_path,
            push_via_bridge=push_via_bridge,
        )
        return path, count

    def transfer_with_report(
        self,
        project: str,
        from_provider: str,
        to_provider: str,
        out_path: Path | None = None,
        push_via_bridge: bool = False,
    ) -> tuple[Path, int, TransferPushReport]:
        """Transfer context and return ``(path, count, push_report)`` for this call."""
        source = from_provider.strip().lower()
        target = to_provider.strip().lower()
        rows = [dict(r) for r in self.store.list_messages_by_provider(project, source)]
        payload = {
            "project": project,
            "from_provider": source,
            "to_provider": target,
            "conversation_id": f"transfer-{source}-to-{target}",
            "messages": [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": row["timestamp"],
                    "metadata": {
                        "origin_provider": row["provider"],
                        "origin_conversation_id": row["conversation_id"],
                        "transfer": True,
                    },
                }
                for row in rows
            ],
        }
        if out_path is None:
            out_dir = Path.home() / ".config" / "deepiri-memorymesh" / "transfers"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{project}.{source}-to-{target}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

        push = TransferPushReport(provider=target)
        if push_via_bridge:
            bridge = Path.home() / ".local" / "bin" / f"memorymesh-bridge-{target}"
            push.attempted = True
            push.bridge_path = bridge
            if not bridge.exists():
                push.success = False
                push.message = f"bridge not found: {bridge}"
                logger.warning(
                    "transfer push failed for provider=%s: %s",
                    target,
                    push.message,
                )
            else:
                try:
                    completed = subprocess.run(
                        [str(bridge), str(out_path)],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                except OSError as exc:
                    push.success = False
                    push.returncode = None
                    push.message = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "transfer push failed for provider=%s bridge=%s: %s",
                        target,
                        bridge,
                        push.message,
                    )
                else:
                    push.returncode = completed.returncode
                    if completed.returncode == 0:
                        push.success = True
                        push.message = "ok"
                    else:
                        push.success = False
                        # Bounded sanitized stderr; never dump payloads/transcripts.
                        raw_err = completed.stderr or ""
                        detail = sanitize_bridge_diagnostic(raw_err)
                        if not detail:
                            detail = f"exit {completed.returncode}"
                        push.message = detail
                        logger.warning(
                            "transfer push failed for provider=%s bridge=%s "
                            "returncode=%s: %s",
                            target,
                            bridge,
                            completed.returncode,
                            push.message,
                        )
        # Compatibility only — threaded callers must use the returned push report.
        self.last_transfer_push = push
        return out_path, len(rows), push
