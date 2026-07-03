from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from .compression import compress_conversation
from .config import Settings
from .embeddings import Embedder
from .models import AgentState, CompressedRecord, MemoryRecord, now_iso
from .packaging import (
    build_udata_payload,
    load_udata,
    transfer_payload_for_provider,
    udata_to_memory_records,
    write_udata_archive,
    write_udata_json,
)
from .providers import parse_provider_file
from .export import (
    ExportFormat,
    copy_to_clipboard,
    gather_project_export,
    normalize_format,
    render_export,
)
from .retrieval import rank_rows
from .scanner import DeviceScanReport, ingest_device, scan_device
from .storage import MemoryStore
from .transfer_delivery import DeliveryResult, deliver_transfer_bundle


class MemoryMesh:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = MemoryStore(settings.db_path)
        self.embedder = Embedder(settings.embedding_backend)

    def init(self) -> None:
        self.store.init()

    def scan_device(self) -> DeviceScanReport:
        return scan_device()

    def ingest_device(
        self,
        project: str,
        providers: list[str] | None = None,
    ) -> DeviceScanReport:
        self.init()
        return ingest_device(project=project, store=self.store, providers=providers)

    def package_udata(
        self,
        project: str,
        output_path: Path,
        ingest_first: bool = True,
        providers: list[str] | None = None,
        compress_after: bool = False,
    ) -> Path:
        """Scan device, optionally ingest, export portable u-data package."""
        self.init()
        scan_report = None
        if ingest_first:
            scan_report = self.ingest_device(project=project, providers=providers)
        else:
            scan_report = scan_device()
        if compress_after:
            self.compress_project(project)
        messages = [dict(r) for r in self.store.list_messages(project)]
        summaries = [dict(r) for r in self.store.list_summaries(project)]
        payload = build_udata_payload(project, messages, summaries, scan=scan_report)
        out = output_path.expanduser()
        if str(out).endswith((".tar.gz", ".tgz")):
            return write_udata_archive(payload, out)
        return write_udata_json(payload, out)

    def import_udata(self, package_path: Path, project_override: str | None = None) -> int:
        payload = load_udata(package_path)
        records = udata_to_memory_records(payload, project_override=project_override)
        return self.store.insert_messages(records)

    def export_provider_transfer(
        self,
        project: str,
        from_provider: str,
        to_provider: str,
        out_path: Path,
    ) -> tuple[Path, int]:
        messages = [dict(r) for r in self.store.list_messages(project)]
        payload = transfer_payload_for_provider(messages, project, from_provider, to_provider)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return out_path, len(payload.get("messages") or [])

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
    ) -> tuple[int, int]:
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        patterns = include_globs or ["*.json", "*.jsonl"]
        files: list[Path] = []
        for pattern in patterns:
            normalized = pattern
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                # For non-recursive mode, drop leading "**/" style hints.
                if normalized.startswith("**/"):
                    normalized = normalized[3:]
                files.extend(directory.glob(normalized))
        inserted = 0
        processed = 0
        for path in sorted(files):
            try:
                inserted += self.ingest_file(provider=provider, project=project, file_path=path)
                processed += 1
            except Exception:
                # Keep sync running even when one export file is malformed.
                continue
        return processed, inserted

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

    def query(self, project: str, text: str, top_k: int = 8) -> list[dict]:
        qvec = self.embedder.embed(text)
        rows = [dict(r) for r in self.store.list_embeddings(project)]
        return rank_rows(qvec, rows, top_k=top_k)

    def export_bundle(self, project: str, output_path: Path) -> Path:
        messages = [dict(r) for r in self.store.list_messages(project)]
        summaries = [dict(r) for r in self.store.list_summaries(project)]
        payload = {
            "project": project,
            "messages": messages,
            "summaries": summaries,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return output_path

    def import_bundle(self, bundle_path: Path, project_override: str | None = None) -> int:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        project = project_override or str(payload.get("project") or "default")
        messages = payload.get("messages") or []
        flattened: list[MemoryRecord] = []
        for msg in messages:
            flattened.append(
                MemoryRecord(
                    provider=str(msg.get("provider") or "bundle"),
                    project=project,
                    conversation_id=str(msg.get("conversation_id") or "bundle"),
                    role=str(msg.get("role") or "unknown"),
                    content=str(msg.get("content") or ""),
                    timestamp=str(msg.get("timestamp") or now_iso()),
                    metadata_json=str(msg.get("metadata_json") or "{}"),
                )
            )
        return self.store.insert_messages(flattened)

    def put_state(self, project: str, agent: str, key: str, value: str) -> None:
        self.store.set_agent_state(
            AgentState(project=project, agent=agent, key=key, value=value)
        )

    def get_state(self, project: str, agent: str, key: str) -> str | None:
        return self.store.get_agent_state(project, agent, key)

    def stats(self, project: str) -> dict[str, int]:
        return self.store.project_stats(project)

    def export_project(
        self,
        project: str,
        fmt: str = "md",
        provider: str | None = None,
        output_path: Path | None = None,
        to_clipboard: bool = False,
    ) -> tuple[str, Path | None, bool]:
        """
        Export all project memory (messages, summaries, agent state) as text.

        Returns (content, written_path_or_none, clipboard_ok).
        """
        export_fmt: ExportFormat = normalize_format(fmt)
        messages = [dict(r) for r in self.store.list_messages(project)]
        summaries = [dict(r) for r in self.store.list_summaries(project)]
        agent_state = [dict(r) for r in self.store.list_agent_state(project)]
        stats = self.stats(project)
        payload = gather_project_export(
            project=project,
            messages=messages,
            summaries=summaries,
            agent_state=agent_state,
            stats=stats,
            provider=provider,
        )
        content = render_export(payload, export_fmt)
        written: Path | None = None
        if output_path is not None:
            out = output_path.expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")
            written = out
        clipboard_ok = False
        if to_clipboard:
            clipboard_ok = copy_to_clipboard(content)
        return content, written, clipboard_ok

    def transfer(
        self,
        project: str,
        from_provider: str,
        to_provider: str,
        out_path: Path | None = None,
        push_via_bridge: bool = False,
        include_summaries: bool = True,
    ) -> tuple[Path, int, DeliveryResult | None]:
        source = from_provider.strip().lower()
        target = to_provider.strip().lower()
        rows = [dict(r) for r in self.store.list_messages_by_provider(project, source)]
        summaries: list[dict[str, str]] = []
        if include_summaries:
            for row in self.store.list_summaries(project):
                summaries.append(
                    {
                        "conversation_id": str(row["conversation_id"]),
                        "summary": str(row["summary"]),
                        "method": str(row["method"]),
                    }
                )
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
            "summaries": summaries,
        }
        if out_path is None:
            out_dir = Path.home() / ".config" / "deepiri-memorymesh" / "transfers"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{project}.{source}-to-{target}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

        delivery: DeliveryResult | None = None
        if push_via_bridge:
            delivery = self.deliver_transfer(out_path, target)
        return out_path, len(rows), delivery

    def deliver_transfer(self, bundle_path: Path, target: str) -> DeliveryResult:
        return deliver_transfer_bundle(bundle_path=bundle_path, target=target, mesh=self)

    def go_transfer(
        self,
        project: str,
        from_provider: str,
        to_provider: str,
        sync_source: bool = True,
        compress_first: bool = True,
        copy_clipboard: bool = True,
    ) -> tuple[Path, DeliveryResult]:
        source = from_provider.strip().lower()
        if sync_source:
            raw = self.settings.provider_paths.get(source, "")
            if raw:
                source_dir = Path(raw).expanduser()
                if source_dir.exists() and source_dir.is_dir():
                    globs = self.settings.provider_globs.get(source, ["**/*.json", "**/*.jsonl"])
                    self.sync_directory(
                        provider=source,
                        project=project,
                        directory=source_dir,
                        recursive=True,
                        include_globs=globs,
                    )
        if compress_first:
            self.compress_project(project)
        bundle_path, _, _ = self.transfer(
            project=project,
            from_provider=source,
            to_provider=to_provider,
            out_path=None,
            push_via_bridge=False,
        )
        delivery = self.deliver_transfer(bundle_path, to_provider.strip().lower())
        if copy_clipboard:
            from .transfer_delivery import try_clipboard_copy

            try_clipboard_copy(delivery.context_md.read_text(encoding="utf-8"))
        return bundle_path, delivery
