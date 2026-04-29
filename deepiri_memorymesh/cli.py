from __future__ import annotations

from pathlib import Path

import typer

from .config import Settings
from .sync_service import MemoryMesh

app = typer.Typer(help="Deepiri MemoryMesh CLI")
state_app = typer.Typer(help="Manage shared agent state")
app.add_typer(state_app, name="state")


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
