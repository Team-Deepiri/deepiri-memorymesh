from __future__ import annotations

from pathlib import Path

import typer

from .config import Settings
from .sync_service import MemoryMesh

app = typer.Typer(help="Deepiri MemoryMesh CLI")
state_app = typer.Typer(help="Manage shared agent state")
bundle_app = typer.Typer(help="Export/import portable context bundles")
app.add_typer(state_app, name="state")
app.add_typer(bundle_app, name="bundle")


def _mesh() -> MemoryMesh:
    settings = Settings.load()
    return MemoryMesh(settings)


@app.command()
def init() -> None:
    """Initialize memory database and default config."""
    mesh = _mesh()
    mesh.init()
    typer.echo(f"Initialized memory store at {mesh.settings.db_path}")


@app.command()
def ingest(
    provider: str = typer.Option(..., help="Provider name: claude/cursor/gemini/etc"),
    project: str = typer.Option(..., help="Project namespace"),
    file: Path = typer.Option(..., exists=True, dir_okay=False, help="Conversation file"),
) -> None:
    """Ingest a conversation export file."""
    mesh = _mesh()
    inserted = mesh.ingest_file(provider=provider, project=project, file_path=file)
    typer.echo(f"Ingested {inserted} message(s) from {file}")


@app.command()
def providers() -> None:
    """List configured providers and default source paths."""
    settings = Settings.load()
    for name in settings.providers:
        path = settings.provider_paths.get(name, "")
        typer.echo(f"{name:16} {path}")


@app.command()
def sync(
    provider: str = typer.Option(..., help="Provider name"),
    project: str = typer.Option(..., help="Project namespace"),
    source_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Directory of exports"),
    recursive: bool = typer.Option(True, help="Recursively scan JSON/JSONL"),
) -> None:
    """Bulk ingest all JSON/JSONL files for a provider."""
    mesh = _mesh()
    processed, inserted = mesh.sync_directory(
        provider=provider,
        project=project,
        directory=source_dir,
        recursive=recursive,
    )
    typer.echo(f"Processed {processed} file(s), inserted {inserted} message(s)")


@app.command("sync-auto")
def sync_auto(
    project: str = typer.Option(..., help="Project namespace"),
    recursive: bool = typer.Option(True, help="Recursively scan JSON/JSONL"),
) -> None:
    """Sync all providers using configured default paths."""
    settings = Settings.load()
    mesh = MemoryMesh(settings)
    total_files = 0
    total_messages = 0
    for provider in settings.providers:
        raw = settings.provider_paths.get(provider, "")
        if not raw:
            continue
        source = Path(raw).expanduser()
        if not source.exists() or not source.is_dir():
            continue
        processed, inserted = mesh.sync_directory(
            provider=provider,
            project=project,
            directory=source,
            recursive=recursive,
        )
        total_files += processed
        total_messages += inserted
        typer.echo(f"{provider}: files={processed} messages={inserted}")
    typer.echo(f"TOTAL files={total_files} messages={total_messages}")


@app.command()
def compress(project: str = typer.Option(..., help="Project namespace")) -> None:
    """Generate compressed memory summaries."""
    mesh = _mesh()
    count = mesh.compress_project(project)
    typer.echo(f"Compressed {count} conversation(s)")


@app.command()
def embed(project: str = typer.Option(..., help="Project namespace")) -> None:
    """Generate embeddings for retrieval."""
    mesh = _mesh()
    count = mesh.embed_project(project)
    typer.echo(f"Embedded {count} message(s)")


@app.command("pipeline")
def pipeline(
    project: str = typer.Option(..., help="Project namespace"),
    auto_sync: bool = typer.Option(False, help="Run sync-auto before compress/embed"),
) -> None:
    """Run end-to-end memory pipeline."""
    mesh = _mesh()
    if auto_sync:
        settings = mesh.settings
        total_files = 0
        total_messages = 0
        for provider in settings.providers:
            source = Path(settings.provider_paths.get(provider, "")).expanduser()
            if not source.exists() or not source.is_dir():
                continue
            processed, inserted = mesh.sync_directory(provider, project, source, recursive=True)
            total_files += processed
            total_messages += inserted
        typer.echo(f"sync-auto: files={total_files} messages={total_messages}")
    summaries = mesh.compress_project(project)
    embeds = mesh.embed_project(project)
    typer.echo(f"pipeline complete: summaries={summaries} embeddings={embeds}")


@app.command()
def query(
    project: str = typer.Option(..., help="Project namespace"),
    q: str = typer.Option(..., help="Search text"),
    top_k: int = typer.Option(8, min=1, max=30),
) -> None:
    """Query memory with semantic retrieval."""
    mesh = _mesh()
    rows = mesh.query(project=project, text=q, top_k=top_k)
    if not rows:
        typer.echo("No results found.")
        raise typer.Exit(0)
    for i, row in enumerate(rows, start=1):
        typer.echo(
            f"[{i}] score={row['score']:.4f} provider={row['provider']} conv={row['conversation_id']}"
        )
        snippet = str(row["content"]).replace("\n", " ")
        typer.echo(f"    {snippet[:220]}")


@bundle_app.command("export")
def bundle_export(
    project: str = typer.Option(...),
    out: Path = typer.Option(..., help="Bundle output path, e.g. ./bundle.json"),
) -> None:
    """Export portable memory bundle."""
    mesh = _mesh()
    path = mesh.export_bundle(project=project, output_path=out)
    typer.echo(f"Exported bundle to {path}")


@bundle_app.command("import")
def bundle_import(
    bundle: Path = typer.Option(..., exists=True, dir_okay=False, help="Bundle JSON path"),
    project: str | None = typer.Option(None, help="Optional project override"),
) -> None:
    """Import portable memory bundle."""
    mesh = _mesh()
    inserted = mesh.import_bundle(bundle_path=bundle, project_override=project)
    typer.echo(f"Imported {inserted} message(s)")


@state_app.command("put")
def state_put(
    project: str = typer.Option(...),
    agent: str = typer.Option(...),
    key: str = typer.Option(...),
    value: str = typer.Option(...),
) -> None:
    """Put shared state key."""
    mesh = _mesh()
    mesh.put_state(project=project, agent=agent, key=key, value=value)
    typer.echo("ok")


@state_app.command("get")
def state_get(
    project: str = typer.Option(...),
    agent: str = typer.Option(...),
    key: str = typer.Option(...),
) -> None:
    """Get shared state key."""
    mesh = _mesh()
    value = mesh.get_state(project=project, agent=agent, key=key)
    if value is None:
        typer.echo("null")
    else:
        typer.echo(value)


if __name__ == "__main__":
    app()
