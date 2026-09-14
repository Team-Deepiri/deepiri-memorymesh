"""Screens for the Memory Mesh session browser (Textual widgets only)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    Label,
    Markdown,
    Select,
    Static,
)

from ..scanner import list_device_sessions
from ..sync_service import MemoryMesh
from .session_model import (
    SessionRow,
    conversation_messages,
    conversation_summary,
    filter_sessions,
    project_stats,
)


@dataclass(slots=True)
class TransferRequest:
    dest: str
    copy_clipboard: bool
    compress_first: bool


@dataclass(slots=True)
class ActionResult:
    name: str
    detail: str


def _when(ts: float) -> str:
    if ts <= 0:
        return "—"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return "—"


def _row_key(row: SessionRow) -> str:
    return f"{row.provider}\x00{row.conversation_id}"


class SessionListScreen(Screen):
    """Main machine-wide session browser."""

    BINDINGS = [
        Binding("/", "focus_filter", "Filter"),
        Binding("enter", "open_detail", "Inspect"),
        Binding("t", "transfer", "Transfer"),
        Binding("a", "actions", "Actions"),
        Binding("?", "help", "Help"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, project: str = "") -> None:
        super().__init__()
        self.project = project
        self.filter_text = ""
        self.provider_filter: str | None = None
        self._rows: list[SessionRow] = []

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static(f"Project: {self.project or '(default)'}", id="project-label"),
            Static("", id="status-label"),
            id="top-bar",
        )
        yield Input(
            placeholder="Filter sessions (provider, workspace, preview, session id)…",
            id="filter",
        )
        yield DataTable(id="sessions")

    def on_mount(self) -> None:
        table = self.query_one("#sessions", DataTable)
        table.cursor_type = "row"
        table.add_columns(
            "Provider",
            "Source",
            "Workspace",
            "Msgs",
            "Last activity",
            "Preview",
        )

    # --- data pipeline -------------------------------------------------

    def refresh_rows(self) -> None:
        rows = filter_sessions(
            self.app.rows, provider=self.provider_filter, text=self.filter_text
        )
        self._rows = rows
        table = self.query_one("#sessions", DataTable)
        table.clear(columns=False)
        for row in rows:
            preview = (row.last_user_preview or "").replace("\n", " ")[:64]
            table.add_row(
                row.provider,
                row.source,
                row.workspace or "—",
                str(row.message_count) if row.message_count else "—",
                _when(row.mtime),
                preview,
                key=_row_key(row),
            )
        self.set_status()

    def set_status(self, extra: str = "") -> None:
        label = self.query_one("#status-label", Static)
        parts = [f"{len(self._rows)} session(s)"]
        if self.provider_filter:
            parts.append(f"provider={self.provider_filter}")
        if self.filter_text:
            parts.append(f"filter={self.filter_text!r}")
        if extra:
            parts.append(extra)
        label.update(" · ".join(parts))

    def selected_row(self) -> SessionRow | None:
        table = self.query_one("#sessions", DataTable)
        idx = table.cursor_row
        if idx is None or not (0 <= idx < len(self._rows)):
            return None
        return self._rows[idx]

    # --- events --------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter":
            self.filter_text = event.value
            self.refresh_rows()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row = self.selected_row()
        if row is not None:
            self.app.show_detail(row)

    # --- actions -------------------------------------------------------

    def action_focus_filter(self) -> None:
        self.query_one("#filter", Input).focus()

    def action_open_detail(self) -> None:
        row = self.selected_row()
        if row is not None:
            self.app.show_detail(row)
        else:
            self.app.notify("No session selected", severity="warning")

    def action_transfer(self) -> None:
        row = self.selected_row()
        if row is not None:
            self.app.show_transfer(row)
        else:
            self.app.notify("Select a session first", severity="warning")

    def action_actions(self) -> None:
        self.app.show_actions(self.project)

    def action_help(self) -> None:
        self.app.show_help()

    def action_refresh(self) -> None:
        self.app.refresh_sessions()


class SessionDetailScreen(ModalScreen[None]):
    """Inspect one conversation: summary + full message transcript."""

    BINDINGS = [
        Binding("escape", "close", "Back"),
        Binding("t", "transfer", "Transfer"),
        Binding("a", "actions", "Actions"),
    ]

    def __init__(self, mesh: MemoryMesh, row: SessionRow, project: str = "") -> None:
        super().__init__()
        self.mesh = mesh
        self.row = row
        self.project = project or row.project

    def compose(self) -> ComposeResult:
        header = (
            f"[b]{self.row.provider}[/b] · {self.row.conversation_id} · "
            f"{self.row.source} · {self.row.message_count} message(s)"
        )
        yield Vertical(
            Static(header, id="detail-header"),
            VerticalScroll(Markdown("_loading transcript…_", id="detail-body")),
            Horizontal(
                Button("Transfer", variant="primary", id="transfer"),
                Button("Export", id="export"),
                Button("Close", id="close"),
                id="detail-actions",
            ),
            id="detail-dialog",
        )

    def on_mount(self) -> None:
        self.run_worker(self._load_transcript(), thread=True, exclusive=True)

    async def _load_transcript(self) -> None:
        summary = conversation_summary(self.mesh, self.row)
        parts: list[str] = []
        if summary:
            parts.append("## Summary\n")
            parts.append(summary)
        messages = conversation_messages(self.mesh, self.row)
        if messages:
            for msg in messages:
                role = str(msg.get("role") or "unknown")
                content = str(msg.get("content") or "").strip() or "(empty)"
                parts.append(f"\n**{role}**: {content}\n")
        else:
            parts.append("_No storable messages found for this session._")
        markdown = "\n".join(parts)
        self.app.call_from_thread(self._render_transcript, markdown)

    def _render_transcript(self, markdown: str) -> None:
        self.query_one("#detail-body", Markdown).update(markdown)

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "close":
            self.dismiss(None)
        elif btn_id == "transfer":
            self.dismiss(None)
            self.app.show_transfer(self.row)
        elif btn_id == "export":
            self.dismiss(None)
            self.app.show_actions(self.project)

    def action_transfer(self) -> None:
        row = self.row
        self.dismiss(None)
        self.app.show_transfer(row)

    def action_actions(self) -> None:
        self.dismiss(None)
        self.app.show_actions(self.project)


class TransferDialog(ModalScreen[TransferRequest | None]):
    """Choose a destination provider and transfer the selected session."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    TITLE = "Transfer session"

    def __init__(
        self,
        mesh: MemoryMesh,
        row: SessionRow,
        project: str = "",
    ) -> None:
        super().__init__()
        self.mesh = mesh
        self.row = row
        self.project = project or row.project
        self.dest: str | None = None

    def compose(self) -> ComposeResult:
        source_line = f"source: [b]{self.row.provider}[/b] ({self.row.source})"
        if self.row.path:
            source_line += f"\nfile: {self.row.path}"
        yield Vertical(
            Label(self.TITLE, classes="dialog-title"),
            Static(source_line, id="transfer-source"),
            Label("Destination provider:"),
            Select(
                options=[(p, p) for p in self.app.transfer_targets(self.row.provider)],
                prompt="Choose a provider…",
                id="dest",
            ),
            Checkbox("Copy context to clipboard after transfer", id="clip"),
            Checkbox("Compress conversation first", id="compress"),
            Horizontal(
                Button("Transfer", variant="primary", id="go"),
                Button("Cancel", id="cancel"),
                id="transfer-buttons",
            ),
            id="transfer-dialog",
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        self.dest = str(event.value) if event.value else None

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id != "go":
            return
        if not self.dest:
            self.app.notify("Choose a destination provider", severity="warning")
            return
        request = TransferRequest(
            dest=self.dest,
            copy_clipboard=self.query_one("#clip", Checkbox).value,
            compress_first=self.query_one("#compress", Checkbox).value,
        )
        self.dismiss(request)


class ActionsScreen(ModalScreen[ActionResult | None]):
    """Project-scoped maintenance actions (stats, compress, embed, scan, sync)."""

    BINDINGS = [Binding("escape", "cancel", "Close")]

    def __init__(self, mesh: MemoryMesh, project: str = "") -> None:
        super().__init__()
        self.mesh = mesh
        self.project = project or "default"

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Project actions", classes="dialog-title"),
            Static(f"project: {self.project}"),
            Button("Project stats", id="stats"),
            Button("Compress conversations (summaries)", id="compress"),
            Button("Embed messages", id="embed"),
            Button("Scan device & ingest", id="scan"),
            Button("Sync-auto configured providers", id="sync"),
            Button("Close", variant="primary", id="close"),
            id="actions-dialog",
        )

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "close":
            self.dismiss(None)
            return
        mapping = {
            "stats": self._run_stats,
            "compress": self._run_compress,
            "embed": self._run_embed,
            "scan": self._run_scan,
            "sync": self._run_sync,
        }
        runner = mapping.get(btn_id)
        if runner is not None:
            self.app.call_from_thread(runner)

    def _run_stats(self) -> None:
        self.dismiss(ActionResult("stats", str(project_stats(self.mesh, self.project))))

    def _run_compress(self) -> None:
        from .session_model import compress_project

        count = compress_project(self.mesh, self.project)
        self.dismiss(ActionResult("compress", f"wrote {count} summary/summaries"))

    def _run_embed(self) -> None:
        from .session_model import embed_project

        count = embed_project(self.mesh, self.project)
        self.dismiss(ActionResult("embed", f"embedded {count} message(s)"))

    def _run_scan(self) -> None:
        report = list_device_sessions()
        self.dismiss(
            ActionResult(
                "scan",
                f"found {len(report)} on-disk session(s) (no ingest yet)",
            )
        )

    def _run_sync(self) -> None:
        from .session_model import sync_all

        report = sync_all(self.mesh, self.project)
        self.dismiss(
            ActionResult(
                "sync",
                f"processed {report.total_processed} · inserted "
                f"{report.total_inserted} · failed {report.total_failed}",
            )
        )


class HelpScreen(ModalScreen[None]):
    """Keyboard reference."""

    BINDINGS = [Binding("escape", "close", "Close"), Binding("?", "close", "Close")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Keyboard reference", classes="dialog-title"),
            Static(
                "\n".join(
                    [
                        "  enter   Inspect the selected session",
                        "  t       Transfer session to another provider",
                        "  a       Project actions (stats / compress / embed / scan)",
                        "  /       Filter sessions",
                        "  r       Rescan device and reload the session list",
                        "  ?       This help",
                        "  q       Quit",
                        "  escape  Back / close dialog",
                    ]
                )
            ),
            Button("Close", variant="primary", id="close"),
            id="help-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


__all__ = [
    "ActionsScreen",
    "ActionResult",
    "HelpScreen",
    "SessionDetailScreen",
    "SessionListScreen",
    "TransferDialog",
    "TransferRequest",
]