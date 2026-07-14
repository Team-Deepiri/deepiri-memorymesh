from pathlib import Path

from ..models import MemoryRecord
from .aider import parse_aider_file
from .base import parse_generic_file
from .claude import parse_claude_file
from .continue_dev import parse_continue_file
from .cursor import parse_cursor_file
from .gemini import parse_gemini_file
from .opencode import parse_opencode_file
from .registry import get_provider, native_parser_map, normalize_provider_key

NATIVE_PROVIDER_PARSERS: dict[str, str] = native_parser_map()


def parse_provider_file(provider: str, project: str, file_path: Path) -> list[MemoryRecord]:
    key = normalize_provider_key(provider)
    cap = get_provider(key)
    if cap.parser_kind == "unsupported":
        # Explicit ingest of unknown/unsupported names still uses the generic
        # parser so deliberate file loads work; sync-auto never routes here.
        return parse_generic_file(key, project, file_path)
    if key in {"claude"}:
        return parse_claude_file(key, project, file_path)
    if key in {"cursor"}:
        return parse_cursor_file(key, project, file_path)
    if key in {"gemini"}:
        return parse_gemini_file(key, project, file_path)
    if key in {"opencode"}:
        return parse_opencode_file(key, project, file_path)
    if key in {"continue"}:
        return parse_continue_file(key, project, file_path)
    if key in {"aider"}:
        return parse_aider_file(key, project, file_path)
    if cap.parser_kind == "generic-explicit":
        return parse_generic_file(key, project, file_path)
    return parse_generic_file(key, project, file_path)


__all__ = ["parse_provider_file", "NATIVE_PROVIDER_PARSERS", "get_provider", "normalize_provider_key"]
