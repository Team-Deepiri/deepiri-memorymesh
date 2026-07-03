from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import typer

from .config import Settings
from .integrations import (
    install_native_integration,
    install_bridge_script,
    install_push_script,
    list_targets,
    write_hook_snippets,
    write_integration_template,
)
from .service_api import run_service
from .sync_service import MemoryMesh
from .tui import run_tui
from .providers import NATIVE_PROVIDER_PARSERS
from .transfer_formats import load_transfer_bundle, render_markdown, render_provider_json
from .transfer_delivery import deliver_transfer_bundle, try_clipboard_copy

app = typer.Typer(help="Deepiri MemoryMesh CLI")
state_app = typer.Typer(help="Manage shared agent state")
bundle_app = typer.Typer(help="Export/import portable context bundles")
package_app = typer.Typer(help="Device scan + portable u-data packaging")
app.add_typer(state_app, name="state")
app.add_typer(bundle_app, name="bundle")
app.add_typer(package_app, name="package")


def _mesh() -> MemoryMesh:
    settings = Settings.load()
    return MemoryMesh(settings)


def _ensure_service_running(host: str = "127.0.0.1", port: int = 8765) -> bool:
    health_url = f"http://{host}:{port}/health"
    try:
        with urlopen(health_url, timeout=0.5) as resp:
            return resp.status == 200
    except URLError:
        pass
    except Exception:
        pass

    subprocess.Popen(
        [sys.executable, "-m", "deepiri_memorymesh.cli", "serve", "--host", host, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(20):
        time.sleep(0.15)
        try:
            with urlopen(health_url, timeout=0.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


@app.command()
def scan(
    ingest: bool = typer.Option(False, "--ingest", help="Ingest discovered data into memory DB"),
    project: str | None = typer.Option(
        None,
        "-p",
        "--project",
        help="Project namespace (required with --ingest)",
    ),
    provider: list[str] = typer.Option(
        [],
        "--provider",
        help="Limit to provider(s): claude, cursor, opencode",
    ),
) -> None:
    """Scan this device for Claude Code, Cursor, and OpenCode conversation data."""
    mesh = _mesh()
    providers = [p.lower() for p in provider] if provider else None
    if ingest:
        if not project:
            typer.echo("error: --project is required when using --ingest")
            raise typer.Exit(1)
        mesh.init()
        report = mesh.ingest_device(project=project, providers=providers)
    else:
        report = mesh.scan_device()
    for line in report.summary_lines():
        typer.echo(line)


@app.command("pull")
def pull(
    project: str = typer.Option(..., "-p", "--project", help="Project namespace"),
    provider: list[str] = typer.Option(
        [],
        "--provider",
        help="Limit to provider(s): claude, cursor, opencode",
    ),
) -> None:
    """Scan device and ingest all Claude/Cursor/OpenCode messages (alias for scan --ingest)."""
    mesh = _mesh()
    mesh.init()
    providers = [p.lower() for p in provider] if provider else None
    report = mesh.ingest_device(project=project, providers=providers)
    for line in report.summary_lines():
        typer.echo(line)


@package_app.command("build")
def package_build(
    project: str = typer.Option(..., "-p", "--project", help="Project namespace"),
    out: Path = typer.Option(
        ...,
        "-o",
        "--out",
        help="Output path (.json or .tar.gz)",
    ),
    no_ingest: bool = typer.Option(False, help="Skip device ingest; export DB only"),
    compress: bool = typer.Option(False, help="Compress conversations before export"),
    provider: list[str] = typer.Option([], "--provider", help="Limit providers"),
) -> None:
    """One-shot: scan device, ingest, export portable u-data package."""
    mesh = _mesh()
    providers = [p.lower() for p in provider] if provider else None
    path = mesh.package_udata(
        project=project,
        output_path=out,
        ingest_first=not no_ingest,
        providers=providers,
        compress_after=compress,
    )
    typer.echo(f"Packaged u-data → {path}")


@package_app.command("import")
def package_import(
    archive: Path = typer.Option(..., exists=True, help="udata .json or .tar.gz"),
    project: str | None = typer.Option(None, "-p", "--project", help="Project override"),
) -> None:
    """Import a portable u-data package from another machine."""
    mesh = _mesh()
    mesh.init()
    count = mesh.import_udata(archive, project_override=project)
    typer.echo(f"Imported {count} message(s)")


@package_app.command("transfer")
def package_transfer(
    project: str = typer.Option(..., "-p", "--project"),
    from_provider: str = typer.Option(..., "--from"),
    to_provider: str = typer.Option(..., "--to"),
    out: Path = typer.Option(..., "-o", "--out"),
) -> None:
    """Export provider-specific transfer JSON for importing into another tool."""
    mesh = _mesh()
    mesh.init()
    path, count = mesh.export_provider_transfer(project, from_provider, to_provider, out)
    typer.echo(f"Wrote {count} message(s) → {path}")


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
def export(
    project: str = typer.Option(..., "-p", "--project", help="Project namespace"),
    format: str = typer.Option(
        "md",
        "--format",
        "-f",
        help="Export format: txt, md (markdown), or json",
    ),
    out: Path | None = typer.Option(
        None,
        "-o",
        "--out",
        help="Write export to this file (prints to stdout if omitted)",
    ),
    clipboard: bool = typer.Option(
        False,
        "--clipboard",
        help="Copy export to system clipboard (wl-copy, xclip, xsel, or pbcopy)",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="Limit export to one provider's messages",
    ),
) -> None:
    """Export all chat/memory for a project as txt, markdown, or JSON."""
    mesh = _mesh()
    mesh.init()
    content, written, clipboard_ok = mesh.export_project(
        project=project,
        fmt=format,
        provider=provider,
        output_path=out,
        to_clipboard=clipboard,
    )
    if written:
        typer.echo(f"Exported → {written}")
    if clipboard:
        if clipboard_ok:
            typer.echo("Copied to clipboard.")
        else:
            typer.echo(
                "warning: could not copy to clipboard "
                "(install wl-clipboard, xclip, or xsel on Linux)",
                err=True,
            )
    if not written and not clipboard:
        typer.echo(content, nl=False)
        if not content.endswith("\n"):
            typer.echo("")


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


@app.command("install-native")
def install_native(
    target: str = typer.Option(..., help="Target app: claude/cursor/gemini/opencode/continue/aider"),
    project: str = typer.Option(..., help="Project namespace"),
    service_url: str = typer.Option("http://127.0.0.1:8765", help="MemoryMesh service URL"),
) -> None:
    """Install native per-provider integration config/plugin/wrapper."""
    paths = install_native_integration(target=target, project=project, service_url=service_url)
    for path in paths:
        typer.echo(f"Wrote {path}")


@app.command("install-native-all")
def install_native_all(
    project: str = typer.Option(..., help="Project namespace"),
    service_url: str = typer.Option("http://127.0.0.1:8765", help="MemoryMesh service URL"),
) -> None:
    """Install native integrations for all supported providers."""
    for target in ["claude", "cursor", "gemini", "opencode", "continue", "aider"]:
        try:
            paths = install_native_integration(target=target, project=project, service_url=service_url)
            typer.echo(f"{target}:")
            for path in paths:
                typer.echo(f"  - {path}")
        except Exception as exc:
            typer.echo(f"{target}: failed ({exc})")


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
        help="Deliver transfer to target inbox and ingest under target provider",
    ),
) -> None:
    """Transfer context from one provider memory layer to another."""
    mesh = _mesh()
    path, count, delivery = mesh.transfer(
        project=project,
        from_provider=from_provider,
        to_provider=to_provider,
        out_path=out,
        push_via_bridge=push,
    )
    typer.echo(f"Transferred {count} message(s) into {path}")
    if delivery:
        typer.echo(f"Delivered inbox: {delivery.inbox_dir}")
        typer.echo(f"Paste file: {delivery.context_md}")
        typer.echo(f"Ingested {delivery.ingested} message(s) under provider={to_provider}")


@app.command("transfer-render")
def transfer_render(
    bundle: Path = typer.Option(..., exists=True, dir_okay=False, help="Transfer bundle JSON"),
    to_provider: str = typer.Option(..., "--to", help="Target provider format"),
    out: Path | None = typer.Option(None, help="Write markdown output path"),
    json_out: Path | None = typer.Option(None, help="Write provider JSON output path"),
) -> None:
    """Render a transfer bundle as paste-ready markdown and/or provider JSON."""
    payload = load_transfer_bundle(bundle)
    md = render_markdown(payload)
    provider_json = render_provider_json(payload, to_provider)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        typer.echo(f"Wrote markdown: {out}")
    else:
        typer.echo(md)
    if json_out:
        import json

        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(provider_json, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Wrote provider JSON: {json_out}")


@app.command("transfer-deliver")
def transfer_deliver(
    bundle: Path = typer.Option(..., exists=True, dir_okay=False, help="Transfer bundle JSON"),
    to_provider: str = typer.Option(..., "--to", help="Target provider"),
    clipboard: bool = typer.Option(False, help="Copy context.md to clipboard if available"),
) -> None:
    """Deliver transfer bundle to target inbox and ingest into MemoryMesh."""
    mesh = _mesh()
    delivery = deliver_transfer_bundle(bundle_path=bundle, target=to_provider, mesh=mesh)
    typer.echo(f"Delivered {delivery.message_count} message(s) to {delivery.inbox_dir}")
    typer.echo(f"context: {delivery.context_md}")
    typer.echo(f"import: {delivery.import_json}")
    typer.echo(f"ingested: {delivery.ingested}")
    if clipboard:
        copied = try_clipboard_copy(delivery.context_md.read_text(encoding="utf-8"))
        typer.echo("clipboard: copied" if copied else "clipboard: unavailable")


@app.command("install-push")
def install_push(
    target: str = typer.Option(..., help="Target provider for push script"),
) -> None:
    """Install memorymesh-push-<target> script for transfer delivery."""
    script_path = install_push_script(target=target)
    typer.echo(f"Installed push script: {script_path}")


@app.command()
def go(
    project: str = typer.Option(..., help="Project namespace"),
    from_provider: str = typer.Option(..., "--from", help="Source provider"),
    to_provider: str = typer.Option(..., "--to", help="Target provider"),
    no_sync: bool = typer.Option(False, help="Skip syncing source provider directory first"),
    no_compress: bool = typer.Option(False, help="Skip compress step before transfer"),
    no_clipboard: bool = typer.Option(False, help="Skip copying context to clipboard"),
) -> None:
    """Full transfer workflow: sync source, compress, bundle, deliver to target inbox."""
    mesh = _mesh()
    bundle_path, delivery = mesh.go_transfer(
        project=project,
        from_provider=from_provider,
        to_provider=to_provider,
        sync_source=not no_sync,
        compress_first=not no_compress,
        copy_clipboard=not no_clipboard,
    )
    typer.echo(f"Bundle: {bundle_path}")
    typer.echo(f"Delivered {delivery.message_count} message(s) to {delivery.inbox_dir}")
    typer.echo(f"Paste into {to_provider}: {delivery.context_md}")
    typer.echo(delivery.instructions_path.read_text(encoding="utf-8"))


@app.command()
def tui(
    project: str | None = typer.Option(
        None,
        help="Project namespace (defaults to current directory name)",
    ),
) -> None:
    """Run interactive MemoryMesh TUI."""
    ok = _ensure_service_running()
    if not ok:
        typer.echo("warning: service did not respond; TUI will still start")
    resolved_project = project or Path.cwd().name or "default"
    run_tui(default_project=resolved_project)


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
