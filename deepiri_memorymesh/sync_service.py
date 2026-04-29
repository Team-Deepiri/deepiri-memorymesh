from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .compression import compress_conversation
from .config import Settings
from .embeddings import Embedder
from .models import AgentState, CompressedRecord
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

    def put_state(self, project: str, agent: str, key: str, value: str) -> None:
        self.store.set_agent_state(
            AgentState(project=project, agent=agent, key=key, value=value)
        )

    def get_state(self, project: str, agent: str, key: str) -> str | None:
        return self.store.get_agent_state(project, agent, key)
