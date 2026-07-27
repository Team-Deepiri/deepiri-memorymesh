"""Portable, deterministic provider file collection (T14)."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterable, Iterator


def _normalize_nonrecursive_pattern(pattern: str) -> str:
    """Strip leading ``**/`` compatibility prefixes for non-recursive scans."""
    normalized = pattern.replace("\\", "/")
    while normalized.startswith("**/"):
        normalized = normalized[3:]
    return normalized


def _pattern_uses_recursive_glob(pattern: str) -> bool:
    return "**" in pattern.replace("\\", "/")


def _pattern_has_directory(pattern: str) -> bool:
    return "/" in pattern.replace("\\", "/")


def _path_passes_through_symlink_dir(root: Path, path: Path) -> bool:
    """True when any directory component under *root* leading to *path* is a symlink."""
    try:
        rel = path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        try:
            rel = path.relative_to(root)
        except ValueError:
            return False
    cur = root
    for part in rel.parts[:-1]:
        cur = cur / part
        try:
            if cur.is_symlink():
                return True
        except OSError:
            return True
    return False


def _iter_nonrecursive_matches(root: Path, pattern: str) -> Iterator[Path]:
    """Match only immediate regular-file children of *root*.

    Patterns that still contain directory components after ``**/`` stripping
    (e.g. ``subdir/*.json``) never traverse; they yield nothing in nonrecursive
    mode.
    """
    normalized = _normalize_nonrecursive_pattern(pattern)
    if not normalized or _pattern_has_directory(normalized):
        return
        yield  # pragma: no cover - makes this a generator
    try:
        children = sorted(root.iterdir(), key=lambda p: str(p))
    except OSError:
        return
    for child in children:
        try:
            # Never descend into symlink directory trees.
            if child.is_dir():
                continue
            if not child.is_file():
                continue
        except OSError:
            continue
        if fnmatch.fnmatch(child.name, normalized):
            yield child


def _iter_recursive_matches(root: Path, pattern: str) -> Iterator[Path]:
    if _pattern_uses_recursive_glob(pattern) or _pattern_has_directory(pattern):
        matches = root.glob(pattern)
    else:
        matches = root.rglob(pattern)
    for path in matches:
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        if _path_passes_through_symlink_dir(root, path):
            continue
        yield path


def collect_provider_files(
    directory: Path,
    patterns: Iterable[str],
    *,
    recursive: bool = True,
) -> list[Path]:
    """Collect regular files matching *patterns* under *directory*.

    Recursive mode
        - Patterns containing ``**`` use :meth:`Path.glob` (already recursive;
          avoids a second ``rglob`` recursion layer).
        - Patterns with an explicit subdirectory (e.g. ``subdir/*.json``) use
          :meth:`Path.glob` so the path meaning is preserved.
        - Simple basename patterns (e.g. ``*.json``) use :meth:`Path.rglob` so
          nested matches are found without requiring a ``**/`` prefix.
        - Paths reached through symlinked directories are excluded.

    Non-recursive mode
        - Leading ``**/`` prefixes are stripped.
        - Only immediate children of *directory* are considered (via
          :meth:`Path.iterdir` + basename ``fnmatch``).
        - Patterns with remaining directory components (e.g. ``subdir/*.json``)
          match nothing — they do not traverse into ``subdir``.
        - Symlinked directory trees are never entered.

    Results are deduplicated (by resolved path when possible), sorted, and
    limited to regular files (directories named ``*.json`` are excluded).

    Compatible with Python 3.10+ pathlib semantics.
    """
    root = Path(directory)
    found: dict[Path, Path] = {}

    for pattern in patterns:
        if not pattern:
            continue
        if recursive:
            matches = _iter_recursive_matches(root, pattern)
        else:
            matches = _iter_nonrecursive_matches(root, pattern)

        for path in matches:
            try:
                key = path.resolve(strict=False)
            except OSError:
                key = path
            existing = found.get(key)
            if existing is None or str(path) < str(existing):
                found[key] = path

    return [found[k] for k in sorted(found.keys(), key=lambda p: str(p))]
