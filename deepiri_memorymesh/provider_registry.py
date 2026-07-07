"""Dynamic provider registry for delivery paths and import instructions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .handoff import HANDOFF_TARGETS, _generic_targets


@dataclass(slots=True)
class ProviderDelivery:
    key: str
    label: str
    handoff_resolver: Callable[[Path], list[Path]]
    import_hint: str


def _hint_paste(target: str) -> str:
    return (
        f"Paste the contents of context.md into a new {target} session "
        f"to continue with transferred context."
    )


def get_provider_delivery(provider: str) -> ProviderDelivery:
    key = provider.strip().lower()
    target = HANDOFF_TARGETS.get(key)
    if target is not None:
        return ProviderDelivery(
            key=target.key,
            label=target.key.title(),
            handoff_resolver=target.resolver,
            import_hint=_hint_paste(target.key),
        )
    return ProviderDelivery(
        key=key,
        label=key.title() or "Agent",
        handoff_resolver=(lambda p: (lambda ws: _generic_targets(p, ws)))(key),
        import_hint=_hint_paste(key or "agent"),
    )


def import_instructions(target: str, inbox_dir: Path) -> str:
    """Provider-aware import instructions (dynamic, not hardcoded per tool)."""
    delivery = get_provider_delivery(target)
    md = inbox_dir / "context.md"
    js = inbox_dir / "import.json"
    return (
        f"MemoryMesh delivered transfer files to:\n"
        f"  - {md}\n"
        f"  - {js}\n\n"
        f"{delivery.import_hint}\n"
        f"Optional: run `memorymesh install-native --target {delivery.key}` for hooks.\n"
    )
