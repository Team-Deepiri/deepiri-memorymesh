"""Workspace path normalization shared across providers."""

from __future__ import annotations

from pathlib import Path


def workspace_slug(workspace: Path) -> str:
    """Map a workspace path to a dashed slug (Claude/Cursor project-dir style)."""
    resolved = workspace.expanduser().resolve()
    slug = str(resolved).replace("/", "-").replace("\\", "-").replace("_", "-")
    if not slug.startswith("-"):
        slug = f"-{slug}"
    return slug
