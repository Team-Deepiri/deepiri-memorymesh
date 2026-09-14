"""Headless smoke tests for the Textual TUI app (no terminal required)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Header

from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.sync_service import MemoryMesh
from deepiri_memorymesh.tui import run_tui
from deepiri_memorymesh.tui.app import MemoryMeshApp
from deepiri_memorymesh.tui.screens import TransferDialog, TransferRequest
from deepiri_memorymesh.tui.session_model import SessionRow


def _mesh(tmp_path: Path) -> MemoryMesh:
    settings = Settings(db_path=tmp_path / "smoke.db", embedding_backend="fallback")
    mesh = MemoryMesh(settings)
    mesh.init()
    return mesh


def test_import_smoke() -> None:
    from deepiri_memorymesh import tui as tui_mod

    assert callable(tui_mod.run_tui)
    assert run_tui.__module__ == "deepiri_memorymesh.tui"


def test_app_boots_and_shows_sessions(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = MemoryMeshApp(_mesh(tmp_path), project="smoke")
        async with app.run_test():
            await asyncio.sleep(0.5)
            assert app.rows == []
            assert app._list_screen is not None
            app.query_one(Header)
            app._list_screen.query_one("#sessions")
            app._sessions_ready([])
            app.show_help()
            await asyncio.sleep(0.1)
            app.pop_screen()
            await asyncio.sleep(0.05)

    asyncio.run(scenario())


def test_transfer_dialog_mounts(tmp_path: Path) -> None:
    from textual.widgets import Select

    async def scenario() -> None:
        app = MemoryMeshApp(_mesh(tmp_path), project="smoke")
        app.rows = [
            SessionRow(
                provider="claude",
                conversation_id="c1",
                source="db",
                project="demo",
                mtime=1.0,
                message_count=2,
            )
        ]
        dialog = TransferDialog(app.mesh, app.rows[0], project="demo")
        async with app.run_test():
            app.push_screen(dialog)
            await asyncio.sleep(0.2)
            dialog.query_one("#dest", Select)
            dialog.query_one("#go")
            dialog.dismiss(
                TransferRequest(dest="opencode", copy_clipboard=False, compress_first=False)
            )
            await asyncio.sleep(0.05)

    asyncio.run(scenario())


def test_transfer_targets_uses_settings(tmp_path: Path) -> None:
    app = MemoryMeshApp(_mesh(tmp_path), project="smoke")
    targets = app.transfer_targets("claude")
    assert "claude" not in targets
    assert "opencode" in targets


def test_failed_transfer_does_not_exit_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated transfer failure")

    async def scenario() -> None:
        app = MemoryMeshApp(_mesh(tmp_path), project="smoke")
        app.rows = [
            SessionRow(
                provider="claude",
                conversation_id="c1",
                source="db",
                project="demo",
                mtime=1.0,
                message_count=2,
            )
        ]
        async with app.run_test() as pilot:
            monkeypatch.setattr(
                "deepiri_memorymesh.tui.session_model.transfer_session", boom
            )
            app._do_transfer(
                app.rows[0],
                TransferRequest(
                    dest="opencode", copy_clipboard=False, compress_first=False
                ),
            )
            await asyncio.sleep(0.4)
            await pilot.pause()
            assert app.is_running

    asyncio.run(scenario())