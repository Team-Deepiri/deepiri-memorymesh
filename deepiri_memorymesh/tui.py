from __future__ import annotations

import curses

from .config import Settings
from .sync_service import MemoryMesh


HELP = [
    "Deepiri MemoryMesh TUI",
    "",
    "[1] Sync (provider -> provider)",
    "[2] Sync All Providers",
    "[3] Compress",
    "[4] Embed",
    "[5] Stats",
    "[6] Query (interactive input)",
    "[h] Provider usage steps",
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


def _provider_steps(provider: str) -> str:
    key = provider.strip().lower()
    if key == "gemini":
        return "Gemini: keep `memorymesh serve` running, then open Gemini and use imported transfer context/session history."
    if key == "opencode":
        return "OpenCode: plugin hook can ingest via bridge; reopen OpenCode if plugin was just installed."
    if key == "cursor":
        return "Cursor: ensure hooks are installed, then run your normal workflow; synced context is available in MemoryMesh."
    if key == "claude":
        return "Claude: hook/ingest uses Claude history/transcripts; continue in Claude and re-sync as needed."
    if key == "continue":
        return "Continue: use hooks config to push events/transcripts via bridge."
    if key == "aider":
        return "Aider: run `aider-memorymesh` wrapper so chat logs are captured automatically."
    return f"{provider}: bridge + transfer bundle path can be used for manual import."


def _pick_provider(stdscr: curses.window, providers: list[str], title: str) -> str | None:
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, title)
        stdscr.addstr(1, 0, "Press number, or ESC to cancel:")
        for idx, provider in enumerate(providers, start=1):
            stdscr.addstr(2 + idx, 0, f"[{idx}] {provider}")
        stdscr.refresh()
        ch = stdscr.getch()
        if ch == 27:  # ESC
            return None
        if ord("1") <= ch <= ord("9"):
            choice = ch - ord("1")
            if 0 <= choice < len(providers):
                return providers[choice]


def run_tui(default_project: str = "deepiri") -> None:
    settings = Settings.load()
    mesh = MemoryMesh(settings)
    mesh.init()

    def _main(stdscr: curses.window) -> None:
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)
        stdscr.timeout(-1)
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        project = default_project
        status = f"Project: {project}"
        detail = ""
        last_key = "none"
        while True:
            stdscr.clear()
            for idx, line in enumerate(HELP):
                stdscr.addstr(idx, 0, line)
            stdscr.addstr(len(HELP) + 1, 0, status[:2000])
            if detail:
                stdscr.addstr(len(HELP) + 2, 0, detail[:2000])
            stdscr.addstr(len(HELP) + 3, 0, f"Last key: {last_key}")
            stdscr.addstr(len(HELP) + 4, 0, "Select option (press key): ")
            stdscr.refresh()
            ch = stdscr.getch()
            last_key = str(ch)
            if ch in (10, 13):  # Enter
                status = "Press a menu key (1-6, p, q)."
                detail = ""
                continue
            if ch in (ord("q"), ord("Q")):
                break
            if ch == ord("1"):
                provider_choices = settings.providers[:9]
                from_p = _pick_provider(stdscr, provider_choices, "Sync FROM provider")
                if not from_p:
                    status = "Sync cancelled"
                    detail = ""
                    continue
                to_p = _pick_provider(stdscr, provider_choices, "Sync TO provider")
                if not to_p:
                    status = "Sync cancelled"
                    detail = ""
                    continue
                status = f"Syncing {from_p} -> {to_p}..."
                detail = "Preparing transfer bundle and pushing via bridge if available..."
                stdscr.addstr(len(HELP) + 1, 0, status[:2000])
                stdscr.addstr(len(HELP) + 2, 0, detail[:2000])
                stdscr.refresh()
                path, count = mesh.transfer(
                    project=project,
                    from_provider=from_p,
                    to_provider=to_p,
                    out_path=None,
                    push_via_bridge=True,
                )
                status = f"Synced {count} message(s) {from_p}->{to_p}"
                detail = f"{_provider_steps(to_p)} | Bundle: {path}"
            elif ch == ord("2"):
                status = "Running sync-auto... (this can take time)"
                detail = "Scanning provider directories..."
                stdscr.addstr(len(HELP) + 1, 0, status[:2000])
                stdscr.addstr(len(HELP) + 2, 0, detail[:2000])
                stdscr.refresh()
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
            elif ch == ord("3"):
                status = "Compressing conversations..."
                detail = ""
                stdscr.addstr(len(HELP) + 1, 0, status[:2000])
                stdscr.refresh()
                count = mesh.compress_project(project)
                status = f"Compressed {count} conversation(s)"
                detail = ""
            elif ch == ord("4"):
                status = "Embedding messages..."
                detail = ""
                stdscr.addstr(len(HELP) + 1, 0, status[:2000])
                stdscr.refresh()
                count = mesh.embed_project(project)
                status = f"Embedded {count} message(s)"
                detail = ""
            elif ch == ord("5"):
                s = mesh.stats(project)
                status = (
                    f"stats messages={s['messages']} conv={s['conversations']} "
                    f"summaries={s['summaries']} emb={s['embeddings']}"
                )
                detail = ""
            elif ch == ord("6"):
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
            elif ch in (ord("h"), ord("H")):
                to_p = _pick_provider(stdscr, settings.providers[:9], "Provider steps for")
                if not to_p:
                    status = "Provider steps cancelled"
                    detail = ""
                else:
                    status = f"Provider guidance: {to_p}"
                    detail = _provider_steps(to_p)
            elif ch in (ord("p"), ord("P")):
                new_project = _readline(stdscr, f"Project [{project}]> ")
                if new_project:
                    project = new_project
                    status = f"Project switched: {project}"
                else:
                    status = f"Project unchanged: {project}"
                detail = ""
            else:
                status = f"Unknown key code: {ch} (use 1-6, h, p, q)"
                detail = ""

    curses.wrapper(_main)
