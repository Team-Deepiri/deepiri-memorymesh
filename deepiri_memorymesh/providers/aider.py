from __future__ import annotations

from pathlib import Path

from ..models import MemoryRecord, now_iso


def parse_aider_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    conv_id = file_path.stem
    text = file_path.read_text(encoding="utf-8")
    records: list[MemoryRecord] = []
    if file_path.suffix.lower() in {".md", ".markdown", ".txt"}:
        # Aider chat logs are often markdown transcripts; keep chunk-level records.
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for chunk in chunks:
            records.append(
                MemoryRecord(
                    provider=provider,
                    project=project,
                    conversation_id=conv_id,
                    role="user_or_assistant",
                    content=chunk,
                    timestamp=now_iso(),
                    metadata_json='{"source":"aider-markdown"}',
                )
            )
        return records
    # Fallback: treat as one blob.
    if text.strip():
        records.append(
            MemoryRecord(
                provider=provider,
                project=project,
                conversation_id=conv_id,
                role="user_or_assistant",
                content=text.strip(),
                timestamp=now_iso(),
                metadata_json='{"source":"aider-generic"}',
            )
        )
    return records
