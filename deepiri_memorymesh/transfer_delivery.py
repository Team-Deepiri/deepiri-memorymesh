from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import MemoryRecord, now_iso
from .transfer_formats import (
    import_instructions,
    load_transfer_bundle,
    messages_from_bundle,
    render_markdown,
    render_provider_json,
)


DEFAULT_INBOX_ROOT = Path.home() / ".config" / "deepiri-memorymesh" / "inbox"


@dataclass(slots=True)
class DeliveryResult:
    inbox_dir: Path
    context_md: Path
    import_json: Path
    transfer_json: Path
    instructions_path: Path
    ingested: int
    message_count: int


def inbox_dir_for(target: str) -> Path:
    return DEFAULT_INBOX_ROOT / target.strip().lower()


def deliver_transfer_bundle(
    bundle_path: Path,
    target: str,
    mesh: object | None = None,
    inbox_root: Path | None = None,
) -> DeliveryResult:
    bundle = load_transfer_bundle(bundle_path)
    key = target.strip().lower()
    bundle_target = str(bundle.get("to_provider") or "").strip().lower()
    if bundle_target and bundle_target != key:
        bundle = dict(bundle)
        bundle["to_provider"] = key

    out_dir = (inbox_root or DEFAULT_INBOX_ROOT) / key
    out_dir.mkdir(parents=True, exist_ok=True)

    context_md = out_dir / "context.md"
    import_json = out_dir / "import.json"
    transfer_json = out_dir / "transfer.json"
    instructions_path = out_dir / "IMPORT_INSTRUCTIONS.txt"

    context_md.write_text(render_markdown(bundle), encoding="utf-8")
    import_json.write_text(
        json.dumps(render_provider_json(bundle, key), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(bundle_path, transfer_json)
    instructions_path.write_text(import_instructions(key, out_dir), encoding="utf-8")

    ingested = 0
    if mesh is not None:
        ingested = _ingest_into_mesh(mesh, bundle, key)

    messages = messages_from_bundle(bundle)
    return DeliveryResult(
        inbox_dir=out_dir,
        context_md=context_md,
        import_json=import_json,
        transfer_json=transfer_json,
        instructions_path=instructions_path,
        ingested=ingested,
        message_count=len(messages),
    )


def _ingest_into_mesh(mesh: object, bundle: dict, target: str) -> int:
    project = str(bundle.get("project") or "default")
    conv_id = str(bundle.get("conversation_id") or f"transfer-{target}")
    records: list[MemoryRecord] = []
    for msg in messages_from_bundle(bundle):
        metadata = msg.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["transfer"] = True
        metadata["origin_provider"] = bundle.get("from_provider")
        records.append(
            MemoryRecord(
                provider=target,
                project=project,
                conversation_id=conv_id,
                role=str(msg["role"]),
                content=str(msg["content"]),
                timestamp=str(msg.get("timestamp") or now_iso()),
                metadata_json=json.dumps(metadata, ensure_ascii=True),
            )
        )
    store = getattr(mesh, "store", None)
    if store is None:
        return 0
    return int(store.insert_messages(records))


def try_clipboard_copy(text: str) -> bool:
    for cmd in (
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["pbcopy"],
        ["wl-copy"],
    ):
        try:
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if proc.returncode == 0:
                return True
        except FileNotFoundError:
            continue
    return False
