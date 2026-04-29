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
    "[5] Query",
    "[6] Transfer (from->to)",
    "[q] Quit",
]


def _prompt(stdscr: curses.window, label: str) -> str:
    curses.echo()
    stdscr.clear()
    stdscr.addstr(0, 0, label)
    stdscr.refresh()
    raw = stdscr.getstr(1, 0, 512)
    curses.noecho()
    return raw.decode("utf-8").strip()


def run_tui(default_project: str = "deepiri") -> None:
    settings = Settings.load()
    mesh = MemoryMesh(settings)
    mesh.init()

    def _main(stdscr: curses.window) -> None:
        project = default_project
        status = f"Project: {project}"
        while True:
            stdscr.clear()
            for idx, line in enumerate(HELP):
                stdscr.addstr(idx, 0, line)
            stdscr.addstr(len(HELP) + 1, 0, status[:200])
            stdscr.addstr(len(HELP) + 3, 0, "Select option: ")
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
            elif key == "2":
                count = mesh.compress_project(project)
                status = f"Compressed {count} conversation(s)"
            elif key == "3":
                count = mesh.embed_project(project)
                status = f"Embedded {count} message(s)"
            elif key == "4":
                s = mesh.stats(project)
                status = (
                    f"stats messages={s['messages']} conv={s['conversations']} "
                    f"summaries={s['summaries']} emb={s['embeddings']}"
                )
            elif key == "5":
                q = _prompt(stdscr, "Enter query text:")
                rows = mesh.query(project, q, top_k=5) if q else []
                if not rows:
                    status = "No query results"
                else:
                    status = "Top: " + " | ".join(
                        f"{r['provider']}:{r['score']:.2f}" for r in rows[:3]
                    )
            elif key == "6":
                from_p = _prompt(stdscr, "From provider:")
                to_p = _prompt(stdscr, "To provider:")
                path, count = mesh.transfer(project, from_p, to_p, out_path=None, push_via_bridge=False)
                status = f"Transferred {count} message(s) -> {path}"
            else:
                status = f"Unknown key: {key}"

    curses.wrapper(_main)
