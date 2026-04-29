from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import subprocess

from .compression import compress_conversation
from .config import Settings
from .embeddings import Embedder
from .models import AgentState, CompressedRecord, MemoryRecord, now_iso
from .providers import parse_provider_file
from .retrieval import rank_rows
from .storage import MemoryStore


class MemoryMesh:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = MemoryStore(settings.db_path)
        self.embedder = Embedder(settings.embedding_backend)

    def init(self) -> None:
        self.store.init()

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
