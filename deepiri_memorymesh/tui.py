from __future__ import annotations

import curses

from .config import Settings
from .sync_service import MemoryMesh


HELP = [
    "Deepiri MemoryMesh TUI",
    "",
    "[1] Sync Auto",
    "[2] Compress",
    "[3] Embed",
    "[4] Stats",
    "[5] Query (interactive input)",
    "[6] Transfer (interactive input)",
    "[p] Change project",
    "[q] Quit",
]


def _readline(stdscr: curses.window, prompt: str) -> str | None:
    """
    Streaming line editor for TUI input.
    Enter submits, Esc cancels.
    """
    buffer: list[str] = []
    while True:
        h, w = stdscr.getmaxyx()
        stdscr.move(h - 2, 0)
        stdscr.clrtoeol()
        line = f"{prompt}{''.join(buffer)}"
        stdscr.addstr(h - 2, 0, line[: max(0, w - 1)])
        stdscr.move(h - 2, min(len(line), max(0, w - 1)))
        stdscr.refresh()

        key = stdscr.get_wch()
        if key == "\n" or key == "\r":
            return "".join(buffer).strip()
        if key == "\x1b":  # ESC
            return None
        if key in ("\b", "\x7f") or key == curses.KEY_BACKSPACE:
            if buffer:
                buffer.pop()
            continue
        if isinstance(key, str) and key.isprintable():
            buffer.append(key)


def run_tui(default_project: str = "deepiri") -> None:
    settings = Settings.load()
    mesh = MemoryMesh(settings)
    mesh.init()

    def _main(stdscr: curses.window) -> None:
        curses.curs_set(1)
        project = default_project
        status = f"Project: {project}"
        detail = ""
        while True:
            stdscr.clear()
            for idx, line in enumerate(HELP):
                stdscr.addstr(idx, 0, line)
            stdscr.addstr(len(HELP) + 1, 0, status[:2000])
            if detail:
                stdscr.addstr(len(HELP) + 2, 0, detail[:2000])
            stdscr.addstr(len(HELP) + 4, 0, "Select option: ")
            stdscr.refresh()
            key = stdscr.getkey()
            if key.lower() == "q":
                break
            if key == "1":
                total_files = 0
                total_messages = 0
                for provider in settings.providers:
                    source = settings.provider_paths.get(provider, "")
                    if not source:
                        continue
                    from pathlib import Path

                    p = Path(source).expanduser()
                    if not p.exists() or not p.is_dir():
                        continue
                    globs = settings.provider_globs.get(provider, ["**/*.json", "**/*.jsonl"])
                    processed, inserted = mesh.sync_directory(
                        provider=provider,
                        project=project,
                        directory=p,
                        recursive=True,
                        include_globs=globs,
                    )
                    total_files += processed
                    total_messages += inserted
                status = f"Sync done: files={total_files}, messages={total_messages}"
                detail = ""
            elif key == "2":
                count = mesh.compress_project(project)
                status = f"Compressed {count} conversation(s)"
                detail = ""
            elif key == "3":
                count = mesh.embed_project(project)
                status = f"Embedded {count} message(s)"
                detail = ""
            elif key == "4":
                s = mesh.stats(project)
                status = (
                    f"stats messages={s['messages']} conv={s['conversations']} "
                    f"summaries={s['summaries']} emb={s['embeddings']}"
                )
                detail = ""
            elif key == "5":
                q = _readline(stdscr, "Query> ")
                rows = mesh.query(project, q, top_k=5) if q else []
                if not rows:
                    status = "No query results"
                    detail = ""
                else:
                    status = "Top providers: " + " | ".join(
                        f"{r['provider']}:{r['score']:.2f}" for r in rows[:3]
                    )
                    detail = str(rows[0]["content"]).replace("\n", " ")[:220]
            elif key == "6":
                from_p = _readline(stdscr, "Transfer from provider> ")
                if not from_p:
                    status = "Transfer cancelled"
                    detail = ""
                    continue
                to_p = _readline(stdscr, "Transfer to provider> ")
                if not to_p:
                    status = "Transfer cancelled"
                    detail = ""
                    continue
                path, count = mesh.transfer(project, from_p, to_p, out_path=None, push_via_bridge=False)
                status = f"Transferred {count} message(s) -> {path}"
                detail = ""
            elif key.lower() == "p":
                new_project = _readline(stdscr, f"Project [{project}]> ")
                if new_project:
                    project = new_project
                    status = f"Project switched: {project}"
                else:
                    status = f"Project unchanged: {project}"
                detail = ""
            else:
                status = f"Unknown key: {key}"
                detail = ""

    curses.wrapper(_main)
