from __future__ import annotations

from pathlib import Path

import typer

from .config import Settings
from .integrations import (
    install_native_integration,
    install_bridge_script,
    list_targets,
    write_hook_snippets,
    write_integration_template,
)
from .service_api import run_service
from .sync_service import MemoryMesh
from .tui import run_tui
from .providers import NATIVE_PROVIDER_PARSERS

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


@app.command("provider-health")
def provider_health() -> None:
    """Show native parser coverage vs fallback providers."""
    settings = Settings.load()
    for name in settings.providers:
        key = name.strip().lower()
        native = "native" if key in NATIVE_PROVIDER_PARSERS else "fallback"
        parser = NATIVE_PROVIDER_PARSERS.get(key, "parse_generic_file")
        typer.echo(f"{key:16} {native:8} {parser}")


@app.command()
def sync(
    provider: str = typer.Option(..., help="Provider name"),
    project: str = typer.Option(..., help="Project namespace"),
    source_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Directory of exports"),
    recursive: bool = typer.Option(True, help="Recursively scan JSON/JSONL"),
) -> None:
    """Bulk ingest all JSON/JSONL files for a provider."""
    mesh = _mesh()
    settings = mesh.settings
    globs = settings.provider_globs.get(provider.strip().lower(), ["**/*.json", "**/*.jsonl"])
    processed, inserted = mesh.sync_directory(
        provider=provider,
        project=project,
        directory=source_dir,
        recursive=recursive,
        include_globs=globs,
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
        globs = settings.provider_globs.get(provider, ["**/*.json", "**/*.jsonl"])
        processed, inserted = mesh.sync_directory(
            provider=provider,
            project=project,
            directory=source,
            recursive=recursive,
            include_globs=globs,
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
            globs = settings.provider_globs.get(provider, ["**/*.json", "**/*.jsonl"])
            processed, inserted = mesh.sync_directory(
                provider, project, source, recursive=True, include_globs=globs
            )
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


@app.command()
def stats(project: str = typer.Option(..., help="Project namespace")) -> None:
    """Show memory layer stats for a project."""
    mesh = _mesh()
    s = mesh.stats(project)
    typer.echo(f"project={project}")
    typer.echo(f"messages={s['messages']}")
    typer.echo(f"conversations={s['conversations']}")
    typer.echo(f"summaries={s['summaries']}")
    typer.echo(f"embeddings={s['embeddings']}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8765, min=1, max=65535, help="Bind port"),
) -> None:
    """Run local MemoryMesh service API for extension/plugin integrations."""
    run_service(host=host, port=port)


@app.command("integrations")
def integrations_list() -> None:
    """List installable code-app integration targets."""
    for target in list_targets():
        typer.echo(f"{target.key:10} {target.extension_hint}")


@app.command("install-integration")
def install_integration(
    target: str = typer.Option(..., help="Target code app: cursor/claude/gemini/opencode/continue"),
    project: str = typer.Option(..., help="Project namespace"),
    service_url: str = typer.Option("http://127.0.0.1:8765", help="MemoryMesh service URL"),
) -> None:
    """Install bridge script + integration template for a code app."""
    script_path = install_bridge_script(target=target, project=project, service_url=service_url)
    template_path = write_integration_template(target=target, project=project)
    typer.echo(f"Installed bridge script: {script_path}")
    typer.echo(f"Wrote integration template: {template_path}")


@app.command("generate-hook-snippets")
def generate_hook_snippets(
    project: str = typer.Option(..., help="Project namespace"),
    out_dir: Path = typer.Option(
        Path("./memorymesh-hooks"),
        help="Directory to write ready-to-paste hook snippets",
    ),
) -> None:
    """Generate ready-to-paste hook configs for supported code apps."""
    files = write_hook_snippets(project=project, output_dir=out_dir)
    for path in files:
        typer.echo(f"Wrote {path}")


@app.command()
def transfer(
    project: str = typer.Option(..., help="Project namespace"),
    from_provider: str = typer.Option(..., "--from", help="Source provider"),
    to_provider: str = typer.Option(..., "--to", help="Target provider"),
    out: Path | None = typer.Option(None, help="Output transfer file path"),
    push: bool = typer.Option(
        False,
        help="Push transfer file to target provider bridge if installed",
    ),
) -> None:
    """Transfer context from one provider memory layer to another."""
    mesh = _mesh()
    path, count = mesh.transfer(
        project=project,
        from_provider=from_provider,
        to_provider=to_provider,
        out_path=out,
        push_via_bridge=push,
    )
    typer.echo(f"Transferred {count} message(s) into {path}")


@app.command()
def tui(project: str = typer.Option("deepiri", help="Default project in TUI")) -> None:
    """Run interactive MemoryMesh TUI."""
    run_tui(default_project=project)


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
