"""The Memory Mesh interactive session browser (Textual App)."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header

from ..scanner import list_device_sessions
from ..sync_service import MemoryMesh
from .screens import (
    ActionsScreen,
    ActionResult,
    HelpScreen,
    SessionDetailScreen,
    SessionListScreen,
    TransferDialog,
    TransferRequest,
)
from .session_model import SessionRow, build_session_list, transfer_destinations


class MemoryMeshApp(App[None]):
    """TUI for browsing, inspecting, and transferring Memory Mesh sessions."""

    TITLE = "Memory Mesh"
    SUB_TITLE = "browse · inspect · transfer"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "show_help", "Help"),
    ]

    def __init__(self, mesh: MemoryMesh, project: str | None = None) -> None:
        super().__init__()
        self.mesh = mesh
        self.project = project or ""
        self.rows: list[SessionRow] = []
        self._list_screen: SessionListScreen | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Footer()

    def on_mount(self) -> None:
        self._list_screen = SessionListScreen(self.project)
        self.push_screen(self._list_screen)
        self.run_worker(
            self._load_sessions(),
            thread=True,
            exclusive=True,
            group="scan",
            exit_on_error=False,
        )

    async def _load_sessions(self) -> None:
        disk = list_device_sessions()
        rows = build_session_list(self.mesh, disk_sessions=disk)
        self.call_from_thread(self._sessions_ready, rows)

    def _sessions_ready(self, rows: list[SessionRow]) -> None:
        self.rows = rows
        if self._list_screen is not None:
            self._list_screen.refresh_rows()

    def refresh_sessions(self) -> None:
        self.run_worker(
            self._load_sessions(),
            thread=True,
            exclusive=True,
            group="scan",
            exit_on_error=False,
        )

    # --- navigation helpers -------------------------------------------

    def show_detail(self, row: SessionRow) -> None:
        self.push_screen(SessionDetailScreen(self.mesh, row, self.project))

    def show_actions(self, project: str) -> None:
        def _on_result(result: ActionResult | None) -> None:
            if result is not None:
                self.notify(f"{result.name}: {result.detail}")
            if self._list_screen is not None:
                self._list_screen.refresh_rows()

        self.push_screen(ActionsScreen(self.mesh, project), callback=_on_result)

    def show_help(self) -> None:
        self.push_screen(HelpScreen())

    def transfer_targets(self, source: str) -> list[str]:
        return transfer_destinations(self.mesh, source)

    def show_transfer(self, row: SessionRow) -> None:
        def _on_request(request: TransferRequest | None) -> None:
            if request is None:
                return
            self._do_transfer(row, request)

        self.push_screen(
            TransferDialog(self.mesh, row, self.project), callback=_on_request
        )

    def _do_transfer(self, row: SessionRow, request: TransferRequest) -> None:
        self.run_worker(
            self._transfer_worker(row, request),
            thread=True,
            exclusive=True,
            group="transfer",
            exit_on_error=False,
        )

    async def _transfer_worker(
        self, row: SessionRow, request: TransferRequest
    ) -> tuple[str, int, Path, bool]:
        from .session_model import transfer_session

        try:
            outcome = transfer_session(
                self.mesh,
                row,
                request.dest,
                project=self.project or row.project,
                copy_clipboard=request.copy_clipboard,
                compress_first=request.compress_first,
            )
        except Exception as exc:
            self.call_from_thread(
                self.notify, f"Transfer failed: {exc}", severity="error"
            )
            return request.dest, 0, Path(), False
        dest = request.dest
        note = " (ingested from disk first)" if outcome.ingested_from_disk else ""
        self.call_from_thread(
            self.notify,
            f"Transferred {outcome.message_count} message(s) → {dest}{note}\n{outcome.bundle_path}",
            timeout=6,
        )
        self.call_from_thread(self.refresh_sessions)
        return (
            dest,
            outcome.message_count,
            outcome.bundle_path,
            outcome.ingested_from_disk,
        )

    def action_show_help(self) -> None:
        self.show_help()