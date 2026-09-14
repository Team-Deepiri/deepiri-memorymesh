"""Interactive Memory Mesh TUI (session browser, inspector, transfer).

The TUI is a thin terminal wrapper over the MemoryMesh SDK. It never starts
detached services and never shells out; all data and transfers go through
:class:`deepiri_memorymesh.sync_service.MemoryMesh`.
"""

from __future__ import annotations

from ..config import Settings
from ..sync_service import MemoryMesh
from .app import MemoryMeshApp


def run_tui(default_project: str = "deepiri") -> None:
    """Launch the interactive session browser (blocking, until the user quits)."""
    settings = Settings.load()
    mesh = MemoryMesh(settings)
    MemoryMeshApp(mesh, project=default_project).run()


__all__ = ["run_tui", "MemoryMeshApp", "MemoryMesh"]