from pathlib import Path

from ..models import MemoryRecord
from .base import parse_generic_file
from .claude import parse_claude_file
from .cursor import parse_cursor_file


def parse_provider_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    key = provider.strip().lower()
    if key in {"claude", "anthropic"}:
        return parse_claude_file(key, project, file_path)
    if key in {"cursor"}:
        return parse_cursor_file(key, project, file_path)
    return parse_generic_file(key, project, file_path)

__all__ = ["parse_provider_file"]
