"""Stable ownership namespaces for the simple Memory facade and legacy import."""

from __future__ import annotations

from dataclasses import dataclass


FACADE_PROVIDER = "python"
FACADE_ROLE = "memory"
SIMPLE_API_CONVERSATION_ID = "python-api"
DEFAULT_PROJECT = "default"

# Legacy import uses the same conversation identity as the public Memory facade
# so imported rows are immediately visible via Memory.all() / Memory.query().
LEGACY_IMPORT_CONVERSATION_ID = SIMPLE_API_CONVERSATION_ID


@dataclass(frozen=True, slots=True)
class MemoryOwnership:
    """Exact ownership filter used by store/query/all and legacy import.

    All four fields participate in facade ownership. ``Memory.query`` and
    ``Memory.all`` intentionally search only this namespace (not every message
    in the project), so they operate over the same logical collection.

    ``import-legacy-memory`` writes into this same namespace for the chosen
    project so migrated simple-memory rows are facade-visible without a
    second lookup path.
    """

    provider: str
    project: str
    conversation_id: str
    role: str


def simple_api_ownership(project: str = DEFAULT_PROJECT) -> MemoryOwnership:
    """Ownership for the public ``Memory`` facade (and legacy import)."""
    return MemoryOwnership(
        provider=FACADE_PROVIDER,
        project=(project.strip() or DEFAULT_PROJECT),
        conversation_id=SIMPLE_API_CONVERSATION_ID,
        role=FACADE_ROLE,
    )


def legacy_import_ownership(project: str = DEFAULT_PROJECT) -> MemoryOwnership:
    """Ownership for ``import-legacy-memory`` — identical to the Memory facade."""
    return simple_api_ownership(project)
