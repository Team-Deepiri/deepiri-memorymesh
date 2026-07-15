from __future__ import annotations

from pathlib import Path
import json
import sys

import typer

from .config import Settings, normalize_db_path
from .integrations import (
    install_native_integration,
    install_bridge_script,
    install_push_script,
    list_targets,
    write_hook_snippets,
    write_integration_template,
)
from .legacy_import import LEGACY_DEFAULT_SOURCE, import_legacy_memory
from .service_api import run_service
from .sync_service import MemoryMesh, SyncDirectoryReport
from .transfer_delivery import deliver_transfer_bundle, try_clipboard_copy
from .transfer_formats import load_transfer_bundle, render_markdown, render_provider_json
from .tui import run_tui
from .providers.registry import list_providers
from .embeddings import Embedder

app = typer.Typer(help="Deepiri MemoryMesh CLI")
state_app = typer.Typer(help="Manage shared agent state")
bundle_app = typer.Typer(help="Export/import portable context bundles")
package_app = typer.Typer(help="Device scan + portable u-data packaging")
conv_app = typer.Typer(help="List and inspect stored conversations")
migrations_app = typer.Typer(help="Schema migration status and apply")
search_index_app = typer.Typer(help="Lexical candidate search-index status/rebuild")
auth_app = typer.Typer(help="Manage HTTP bearer-token authentication (T37)")
auth_token_app = typer.Typer(help="Create/list/revoke/rotate project-scoped tokens")
pull_app = typer.Typer(help="Pull memory from device scan or token-gated external sources")
encryption_app = typer.Typer(help="Optional encryption at rest (T33; requires memorymesh[security])")
app.add_typer(state_app, name="state")
app.add_typer(bundle_app, name="bundle")
app.add_typer(package_app, name="package")
app.add_typer(conv_app, name="conversations")
app.add_typer(migrations_app, name="migrations")
app.add_typer(search_index_app, name="search-index")
app.add_typer(auth_app, name="auth")
auth_app.add_typer(auth_token_app, name="token")
app.add_typer(pull_app, name="pull")
app.add_typer(encryption_app, name="encryption")


def _mesh() -> MemoryMesh:
    settings = Settings.load()
    return MemoryMesh(settings)


def _report_sync_failures(report: SyncDirectoryReport, *, prefix: str = "") -> None:
    """Print per-file sync failures to stderr without full tracebacks."""
    label = f"{prefix}: " if prefix else ""
    for failure in report.failures:
        typer.echo(
            f"{label}FAILED {failure.path}: {failure.error_type}: {failure.message}",
            err=True,
        )


def _echo_sync_summary(
    report: SyncDirectoryReport,
    *,
    prefix: str | None = None,
) -> None:
    head = f"{prefix}: " if prefix else ""
    if report.failed:
        typer.echo(
            f"{head}Processed {report.processed} file(s), "
            f"failed {report.failed}, inserted {report.inserted} message(s)",
            err=True,
        )
    else:
        typer.echo(
            f"{head}Processed {report.processed} file(s), inserted {report.inserted} message(s)"
        )


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


@pull_app.command("device")
def pull_device(
    project: str = typer.Option(..., "-p", "--project", help="Project namespace"),
    provider: list[str] = typer.Option(
        [],
        "--provider",
        help="Limit to provider(s): claude, cursor, opencode",
    ),
) -> None:
    """Scan device and ingest Claude/Cursor/OpenCode messages (alias for scan --ingest)."""
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
    """Show provider capability registry status (T19)."""
    settings = Settings.load()
    configured = {p.strip().lower() for p in settings.providers}
    for cap in list_providers():
        key = cap.name
        in_cfg = "configured" if key in configured else "default-registry"
        auto = "auto" if cap.automatic_discovery else "no-auto"
        path = settings.provider_paths.get(key) or cap.default_path or ""
        evidence = ",".join(cap.evidence) if cap.evidence else "none"
        typer.echo(
            f"{key:16} kind={cap.parser_kind:16} {auto:7} {in_cfg:16} "
            f"integration={cap.integration_support} evidence={evidence} path={path}"
        )
        if key in configured and cap.parser_kind == "unsupported":
            typer.echo(
                f"{'':16} note=configured-but-skipped-by-sync-auto "
                f"limitations={cap.limitations}",
                err=True,
            )


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
    report = mesh.sync_directory_report(
        provider=provider,
        project=project,
        directory=source_dir,
        recursive=recursive,
        include_globs=globs,
    )
    _report_sync_failures(report)
    _echo_sync_summary(report)


@app.command("sync-auto")
def sync_auto(
    project: str = typer.Option(..., help="Project namespace"),
    recursive: bool = typer.Option(True, help="Recursively scan JSON/JSONL"),
) -> None:
    """Sync auto-discoverable providers using configured default paths."""
    settings = Settings.load()
    mesh = MemoryMesh(settings)
    report = mesh.sync_auto_report(project=project, recursive=recursive)
    for outcome in report.outcomes:
        if outcome.skipped:
            typer.echo(
                f"{outcome.provider}: SKIPPED ({outcome.classification}) {outcome.reason}",
                err=True,
            )
            continue
        if outcome.failed:
            typer.echo(
                f"{outcome.provider}: files={outcome.processed} failed={outcome.failed} "
                f"messages={outcome.inserted}",
                err=True,
            )
        else:
            typer.echo(
                f"{outcome.provider}: files={outcome.processed} messages={outcome.inserted}"
            )
    if report.total_failed or report.skipped_unsupported:
        typer.echo(
            f"TOTAL files={report.total_processed} failed={report.total_failed} "
            f"messages={report.total_inserted} "
            f"skipped_unsupported={report.skipped_unsupported} "
            f"skipped_missing_path={report.skipped_missing_path}",
            err=True,
        )
    else:
        typer.echo(
            f"TOTAL files={report.total_processed} messages={report.total_inserted}"
        )


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
        auto_report = mesh.sync_auto_report(project=project, recursive=True)
        for outcome in auto_report.outcomes:
            if outcome.skipped:
                typer.echo(
                    f"{outcome.provider}: SKIPPED ({outcome.classification}) {outcome.reason}",
                    err=True,
                )
        if auto_report.total_failed or auto_report.skipped_unsupported:
            typer.echo(
                f"sync-auto: files={auto_report.total_processed} "
                f"failed={auto_report.total_failed} "
                f"messages={auto_report.total_inserted}",
                err=True,
            )
        else:
            typer.echo(
                f"sync-auto: files={auto_report.total_processed} "
                f"messages={auto_report.total_inserted}"
            )
    summaries = mesh.compress_project(project)
    embeds = mesh.embed_project(project)
    typer.echo(f"pipeline complete: summaries={summaries} embeddings={embeds}")


@app.command()
def query(
    project: str = typer.Option(..., help="Project namespace"),
    q: str = typer.Option(..., help="Search text"),
    top_k: int = typer.Option(8, min=1, max=30),
    mode: str = typer.Option(
        "auto",
        help="Retrieval mode: exact | indexed | auto",
    ),
    candidate_limit: int | None = typer.Option(
        None,
        help="Max lexical candidates when using indexed/auto (default from settings)",
    ),
) -> None:
    """Query memory with semantic retrieval."""
    mesh = _mesh()
    result = mesh.query_with_report(
        project=project,
        text=q,
        top_k=top_k,
        strategy=mode,
        candidate_limit=candidate_limit,
    )
    report = result.report
    typer.echo(
        f"strategy_requested={report.strategy_requested} "
        f"strategy_used={report.strategy_used} "
        f"eligible={report.total_eligible_embeddings} "
        f"candidates={report.candidate_message_count} "
        f"scored={report.embeddings_scored}",
        err=True,
    )
    if report.exact_fallback_reason:
        typer.echo(f"exact_fallback_reason={report.exact_fallback_reason}", err=True)
    diag = result.diagnostic
    if diag:
        typer.echo(diag, err=True)
    rows = result.rows
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


@app.command("embedding-status")
def embedding_status() -> None:
    """Show requested vs active embedding backend (T30)."""
    mesh = _mesh()
    status = mesh.embedding_status()
    typer.echo(f"requested_backend={status.requested_backend}")
    typer.echo(f"active_backend={status.active_backend}")
    typer.echo(f"model={status.model or ''}")
    typer.echo(f"dimensions={status.dimensions if status.dimensions is not None else ''}")
    typer.echo(f"fallback_occurred={str(status.fallback_occurred).lower()}")
    if status.fallback_reason:
        typer.echo(f"fallback_reason={status.fallback_reason}")
    typer.echo(f"stable_backend_id={status.stable_backend_id}")


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        help="Bind host (loopback only: 127.0.0.1, localhost, or ::1)",
    ),
    port: int = typer.Option(8765, min=1, max=65535, help="Bind port"),
    ingest_root: list[Path] = typer.Option(
        [],
        "--ingest-root",
        help="Additional allowed root directory for HTTP file_path ingest (repeatable)",
    ),
    auth_mode: str | None = typer.Option(
        None,
        "--auth-mode",
        help="required|off (default: required, or Settings.http_auth_mode)",
    ),
) -> None:
    """Run local MemoryMesh service API for extension/plugin integrations."""
    from .http_security import assert_loopback_host, normalize_ingest_roots
    from .service_api import AUTH_MODES

    try:
        assert_loopback_host(host)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    settings = Settings.load()
    resolved_auth_mode = auth_mode or settings.http_auth_mode or "required"
    if resolved_auth_mode not in AUTH_MODES:
        typer.echo(
            f"error: --auth-mode must be one of {sorted(AUTH_MODES)}, got {resolved_auth_mode!r}",
            err=True,
        )
        raise typer.Exit(code=1)
    if resolved_auth_mode == "off":
        typer.echo(
            "WARNING: starting memorymesh service with authentication DISABLED "
            "(--auth-mode off). Any local process/user can read and write every "
            "project via this HTTP API. Use only for local development.",
            err=True,
        )
    roots = normalize_ingest_roots(
        provider_paths=settings.provider_paths,
        extra_roots=list(ingest_root),
    )
    run_service(host=host, port=port, ingest_roots=roots, settings=settings, auth_mode=resolved_auth_mode)


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
    token_file: Path | None = typer.Option(
        None,
        "--token-file",
        help="Path to HTTP project token file (never embedded in generated scripts)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only; write nothing"),
    update: bool = typer.Option(
        False,
        "--update",
        help="Replace an existing differing installation for this target",
    ),
    auth_required: bool = typer.Option(
        True,
        "--auth-required/--no-auth-required",
        help="Require a token-file when HTTP auth is expected (default: require)",
    ),
) -> None:
    """Install native per-provider integration (transactional; T12)."""
    from .integration_install import install_native_transactional

    report = install_native_transactional(
        target=target,
        project=project,
        service_url=service_url,
        token_file=token_file,
        dry_run=dry_run,
        update=update,
        auth_required=auth_required,
    )
    if report.dry_run and report.plan is not None:
        typer.echo(f"dry-run provider={report.plan.provider}")
        for action in report.plan.actions:
            typer.echo(f"  action: {action}")
        for path in report.plan.would_write:
            typer.echo(f"  would_write: {path}")
        return
    if report.noop:
        typer.echo(report.message)
        return
    if not report.ok:
        typer.echo(f"error: {report.message}", err=True)
        for err in report.rollback_errors:
            typer.echo(f"  rollback: {err}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"installation_id={report.installation_id}")
    if report.manifest_path is not None:
        typer.echo(f"manifest={report.manifest_path}")


@app.command("install-native-all")
def install_native_all(
    project: str = typer.Option(..., help="Project namespace"),
    service_url: str = typer.Option("http://127.0.0.1:8765", help="MemoryMesh service URL"),
    token_file: Path | None = typer.Option(None, "--token-file"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    update: bool = typer.Option(False, "--update"),
) -> None:
    """Install native integrations for all registry-approved native providers."""
    from .integration_install import install_native_transactional
    from .providers.registry import native_integration_provider_names

    any_failed = False
    for target in native_integration_provider_names():
        report = install_native_transactional(
            target=target,
            project=project,
            service_url=service_url,
            token_file=token_file,
            dry_run=dry_run,
            update=update,
            auth_required=token_file is not None,
        )
        if report.ok:
            typer.echo(f"{target}: ok ({report.message})")
        else:
            any_failed = True
            typer.echo(f"{target}: FAILED ({report.message})", err=True)
    if any_failed:
        raise typer.Exit(code=1)


@app.command("uninstall-native")
def uninstall_native(
    target: str = typer.Option(..., help="Target app to uninstall"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Remove generated files even if modified (still never removes unrelated paths)",
    ),
) -> None:
    """Remove Memory Mesh-owned native integration artifacts (T36)."""
    from .integration_install import uninstall_native_transactional

    report = uninstall_native_transactional(target, dry_run=dry_run, force=force)
    if report.dry_run:
        typer.echo(f"dry-run uninstall {target}")
        for path in report.removed_files:
            typer.echo(f"  would_remove: {path}")
        return
    typer.echo(report.message)
    for path in report.removed_files:
        typer.echo(f"  removed: {path}")
    for path in report.preserved_files:
        typer.echo(f"  preserved: {path}")
    for conflict in report.conflicts:
        typer.echo(f"  conflict: {conflict}", err=True)


@app.command("uninstall-native-all")
def uninstall_native_all(
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Uninstall all recorded Memory Mesh native installations."""
    from .integration_install import list_installations, uninstall_native_transactional

    for manifest in list_installations():
        report = uninstall_native_transactional(
            manifest.provider, dry_run=dry_run, force=force, installation_id=manifest.installation_id
        )
        typer.echo(f"{manifest.provider}/{manifest.installation_id}: {report.message}")


integrations_mgmt_app = typer.Typer(help="Integration installation status, verify, restore")
app.add_typer(integrations_mgmt_app, name="integrations-mgmt")


@integrations_mgmt_app.command("status")
def integrations_status_cmd() -> None:
    """List recorded native installations (no secrets)."""
    from .integration_install import list_installations

    rows = list_installations()
    if not rows:
        typer.echo("No recorded installations.")
        return
    for m in rows:
        typer.echo(
            f"id={m.installation_id} provider={m.provider} project={m.project} "
            f"token_file={m.token_file or ''} files={len(m.generated_files)}"
        )


@integrations_mgmt_app.command("verify")
def integrations_verify_cmd(
    installation_id: str | None = typer.Option(None, "--installation-id"),
) -> None:
    """Verify installed artifacts against manifests."""
    from .integration_install import list_installations, verify_installation

    ids = (
        [installation_id]
        if installation_id
        else [m.installation_id for m in list_installations()]
    )
    if not ids:
        typer.echo("No installations to verify.")
        return
    failed = False
    for iid in ids:
        result = verify_installation(iid)
        typer.echo(json.dumps(result, ensure_ascii=True))
        if not result.get("ok"):
            failed = True
    if failed:
        raise typer.Exit(code=1)


@integrations_mgmt_app.command("restore")
def integrations_restore_cmd(
    installation_id: str = typer.Option(..., "--installation-id"),
    yes: bool = typer.Option(False, "--yes", help="Confirm restore from install backup"),
) -> None:
    """Restore files from an installation's recorded backups."""
    from .integration_install import restore_from_backup

    result = restore_from_backup(installation_id, yes=yes)
    typer.echo(json.dumps(result, ensure_ascii=True))
    if not result.get("ok"):
        raise typer.Exit(code=1)

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
    conversation: str | None = typer.Option(
        None,
        "-c",
        "--conversation",
        help="Limit transfer to one conversation id (substring match)",
    ),
) -> None:
    """Transfer context from one provider memory layer to another."""
    mesh = _mesh()
    path, count, push_report = mesh.transfer_with_report(
        project=project,
        from_provider=from_provider,
        to_provider=to_provider,
        out_path=out,
        push_via_bridge=push,
        conversation_id=conversation,
    )
    typer.echo(f"Transferred {count} message(s) into {path}")
    if push and push_report.attempted:
        if push_report.success:
            typer.echo(f"Push ok via {push_report.bridge_path}")
        else:
            typer.echo(
                f"Push failed for {push_report.provider}: {push_report.message}",
                err=True,
            )
            raise typer.Exit(code=1)


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
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(provider_json, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
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
    conversation: str | None = typer.Option(
        None,
        "-c",
        "--conversation",
        help="Limit transfer to one conversation id (substring match)",
    ),
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
        conversation_id=conversation,
    )
    typer.echo(f"Bundle: {bundle_path}")
    typer.echo(f"Delivered {delivery.message_count} message(s) to {delivery.inbox_dir}")
    typer.echo(f"Paste into {to_provider}: {delivery.context_md}")
    typer.echo(delivery.instructions_path.read_text(encoding="utf-8"))


@conv_app.command("list")
def conversations_list(
    project: str = typer.Option(..., "-p", "--project", help="Project namespace"),
    provider: str | None = typer.Option(None, "--provider", help="Filter by provider"),
    limit: int = typer.Option(20, min=1, max=100),
) -> None:
    """List stored conversations (newest first) with message counts and previews."""
    mesh = _mesh()
    rows = mesh.store.list_conversations(project=project, provider=provider, limit=limit)
    if not rows:
        typer.echo(f"No conversations for project={project}")
        raise typer.Exit(0)
    for row in rows:
        preview = str(row["last_user_preview"] or "").replace("\n", " ")[:120]
        typer.echo(
            f"{row['conversation_id']}\t"
            f"provider={row['provider']}\t"
            f"msgs={row['message_count']}\t"
            f"last={row['last_timestamp']}"
        )
        if preview:
            typer.echo(f"  preview: {preview}")


@app.command()
def resume(
    project: str = typer.Option(..., "-p", "--project", help="Project namespace"),
    from_provider: str = typer.Option(..., "--from", help="Source provider"),
    to_provider: str = typer.Option(..., "--to", help="Target provider"),
    workspace: Path | None = typer.Option(
        None,
        "-w",
        "--workspace",
        help="Workspace to correlate sessions (default: cwd)",
    ),
    conversation: str | None = typer.Option(
        None,
        "-c",
        "--conversation",
        help="Optional conversation id; auto-detected from workspace when omitted",
    ),
    no_sync: bool = typer.Option(False, help="Skip syncing source provider first"),
    clipboard: bool = typer.Option(True, help="Copy resume brief to clipboard"),
) -> None:
    """Workspace Session Bridge — auto-pick source session and resume in target agent."""
    mesh = _mesh()
    ws = (workspace or Path.cwd()).expanduser().resolve()
    try:
        bundle_path, delivery, meta = mesh.resume_session(
            project=project,
            from_provider=from_provider,
            to_provider=to_provider,
            workspace=ws,
            conversation_id=conversation,
            sync_source=not no_sync,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}")
        raise typer.Exit(1) from exc

    typer.echo(f"Resolved session: {meta.get('resolved_conversation_id')}")
    typer.echo(f"Transferred {meta.get('message_count')} message(s)")
    typer.echo(f"Bundle: {bundle_path}")
    typer.echo(f"Inbox: {delivery.context_md}")
    for path in meta.get("handoff_files") or []:
        typer.echo(f"Handoff: {path}")
    if clipboard:
        copied = try_clipboard_copy(delivery.context_md.read_text(encoding="utf-8"))
        typer.echo("clipboard: copied" if copied else "clipboard: unavailable")
    typer.echo(
        f"\nTarget agent ({to_provider}) can read the handoff file(s) above "
        f"or inbox context.md to continue the session."
    )


@app.command()
def handoff(
    project: str = typer.Option(..., "-p", "--project", help="Project namespace"),
    from_provider: str = typer.Option(..., "--from", help="Source provider"),
    to_provider: str = typer.Option(..., "--to", help="Target provider"),
    conversation: str | None = typer.Option(
        None,
        "-c",
        "--conversation",
        help="Conversation id (substring). Auto-detected when omitted.",
    ),
    workspace: Path | None = typer.Option(
        None,
        "-w",
        "--workspace",
        help="Workspace directory for provider handoff files (default: cwd)",
    ),
    no_sync: bool = typer.Option(False, help="Skip syncing source provider first"),
    clipboard: bool = typer.Option(True, help="Copy handoff markdown to clipboard"),
) -> None:
    """Alias for resume — transfer one session with workspace-aware handoff files."""
    resume(
        project=project,
        from_provider=from_provider,
        to_provider=to_provider,
        workspace=workspace,
        conversation=conversation,
        no_sync=no_sync,
        clipboard=clipboard,
    )


@app.command()
def tui(
    project: str | None = typer.Option(
        None,
        help="Project namespace (defaults to current directory name)",
    ),
    with_service: bool = typer.Option(
        False,
        "--with-service",
        help="Start a supervised in-process HTTP service owned by this TUI",
    ),
    host: str = typer.Option("127.0.0.1", help="Loopback host for optional service"),
    port: int = typer.Option(8765, min=1, max=65535, help="Port for optional service"),
) -> None:
    """Run interactive MemoryMesh TUI (direct local; no detached server)."""
    from .supervised_service import SupervisedService, detect_existing_service

    resolved_project = project or Path.cwd().name or "default"
    settings = Settings.load()
    supervised: SupervisedService | None = None
    try:
        if with_service:
            supervised = SupervisedService(host=host, port=port, settings=settings)
            result = supervised.start()
            if result in {"port_conflict", "port_conflict_unrelated", "start_failed"}:
                typer.echo(
                    f"error: could not start owned service ({result}); "
                    "refusing to kill an unrelated process on the port",
                    err=True,
                )
                raise typer.Exit(code=1)
            if supervised.reused_existing:
                typer.echo(f"reusing compatible existing service on {host}:{port}")
            else:
                typer.echo(f"started supervised service on {host}:{port}")
        else:
            existing = detect_existing_service(host, port)
            if existing.ok and existing.compatible:
                typer.echo(
                    f"note: compatible service already running on {host}:{port} "
                    "(not owned by this TUI)"
                )
            elif existing.ok and not existing.compatible:
                typer.echo(
                    f"warning: unrelated HTTP service on {host}:{port}; "
                    "TUI continues without owning it",
                    err=True,
                )
        run_tui(default_project=resolved_project)
    finally:
        if supervised is not None:
            supervised.shutdown()


@bundle_app.command("export")
def bundle_export(
    project: str = typer.Option(...),
    out: Path = typer.Option(..., help="Bundle output path, e.g. ./bundle.json"),
    allow_plaintext: bool = typer.Option(
        False,
        "--allow-plaintext",
        help="When DB encryption is enabled, allow writing an unencrypted bundle (warns)",
    ),
) -> None:
    """Export portable memory bundle.

    When database encryption is enabled, the default is an encrypted bundle
    envelope using the active key. Pass --allow-plaintext to emit plaintext
    (explicit opt-in; prints a warning).
    """
    mesh = _mesh()
    path = mesh.export_bundle(
        project=project,
        output_path=out,
        allow_plaintext=allow_plaintext,
    )
    typer.echo(f"Exported bundle to {path}")


@bundle_app.command("import")
def bundle_import(
    bundle: Path = typer.Option(..., exists=True, dir_okay=False, help="Bundle JSON path"),
    project: str | None = typer.Option(None, help="Optional project override"),
) -> None:
    """Import portable memory bundle (messages and summaries).

    Accepts plaintext bundles and encrypted Memory Mesh bundle envelopes.
    """
    mesh = _mesh()
    report = mesh.import_bundle_with_report(
        bundle_path=bundle, project_override=project
    )
    typer.echo(
        f"Imported messages={report.messages_inserted}/{report.messages_seen} "
        f"(duplicates={report.messages_duplicate}) "
        f"summaries_inserted={report.summaries_inserted} "
        f"summaries_updated={report.summaries_updated} "
        f"summaries_seen={report.summaries_seen}"
    )
    if report.malformed_messages or report.malformed_summaries:
        typer.echo(
            f"Skipped malformed messages={report.malformed_messages} "
            f"summaries={report.malformed_summaries}",
            err=True,
        )


@app.command("import-legacy-memory")
def import_legacy_memory_cmd(
    source: Path = typer.Option(
        LEGACY_DEFAULT_SOURCE,
        help="Legacy simple-memory SQLite path (default: ~/.memorymesh/memory.db)",
    ),
    project: str = typer.Option("default", help="Destination project namespace"),
    destination: Path | None = typer.Option(
        None,
        help="Destination platform database (default: configured/canonical path)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Inspect and report without writing to the destination",
    ),
) -> None:
    """Import the historical simple Memory database into the platform schema.

    Never modifies the source file. Re-embeds with the active configured backend.
    """
    from .config import default_config_path, default_db_path

    cfg_path = default_config_path()
    if cfg_path.exists():
        settings = Settings.load()
        backend = settings.embedding_backend
        default_dest = settings.db_path
    else:
        # Do not create YAML merely to run the importer.
        backend = "fallback"
        default_dest = default_db_path()
    dest = normalize_db_path(destination) if destination is not None else default_dest
    embedder = Embedder(backend)
    try:
        report = import_legacy_memory(
            source=source,
            destination=dest,
            project=project,
            dry_run=dry_run,
            embedder=embedder,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    mode = "dry-run" if report.dry_run else "import"
    typer.echo(f"Legacy {mode}: source={report.source}")
    typer.echo(f"destination={report.destination} project={report.project}")
    if report.dry_run:
        typer.echo(
            f"Would import: scanned={report.scanned} importable={report.importable} "
            f"imported={report.imported} duplicates_skipped={report.duplicates_skipped} "
            f"failed={report.failed}"
        )
        typer.echo("No changes were made")
    else:
        typer.echo(
            f"scanned={report.scanned} importable={report.importable} "
            f"imported={report.imported} duplicates_skipped={report.duplicates_skipped} "
            f"failed={report.failed}"
        )
    typer.echo(report.warning)
    if report.failed:
        for failure in report.failures[:20]:
            typer.echo(
                f"  rowid={failure.rowid}: {failure.error_type}: {failure.message}",
                err=True,
            )


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


@migrations_app.command("status")
def migrations_status() -> None:
    """Show schema migration status (read-only)."""
    from .migrations import migration_status

    mesh = _mesh()
    status = migration_status(mesh.settings.db_path)
    typer.echo(f"db_path={status.db_path}")
    typer.echo(f"current_version={status.current_version}")
    typer.echo(f"latest_version={status.latest_version}")
    if status.journal_mode:
        typer.echo(f"journal_mode={status.journal_mode}")
    if status.adopted_unversioned_baseline:
        typer.echo("unversioned_batch4_baseline=yes")
    if not status.pending:
        typer.echo("pending=none")
    else:
        for version, name in status.pending:
            typer.echo(f"pending={version}:{name}")


@migrations_app.command("apply")
def migrations_apply(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report pending migrations without writing or creating a backup",
    ),
) -> None:
    """Apply pending schema migrations."""
    from .migrations import MigrationError, migrate

    mesh = _mesh()
    try:
        report = migrate(mesh.settings.db_path, dry_run=dry_run)
    except MigrationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"db_path={report.db_path}")
    typer.echo(f"from_version={report.from_version}")
    typer.echo(f"to_version={report.to_version}")
    if report.dry_run:
        typer.echo("dry_run=yes")
        typer.echo("No changes were made")
        for version, name in report.pending:
            typer.echo(f"pending={version}:{name}")
        return
    if report.no_change:
        typer.echo("no_change=yes")
        return
    for version, name in report.applied:
        typer.echo(f"applied={version}:{name}")
    if report.backup_path:
        typer.echo(f"backup_path={report.backup_path}")
    if report.adopted_unversioned_baseline:
        typer.echo("adopted_unversioned_baseline=yes")


@app.command("database-status")
def database_status() -> None:
    """Show schema version, journal mode, foreign keys, and busy timeout."""
    mesh = _mesh()
    mesh.init()
    status = mesh.store.database_status()
    typer.echo(f"db_path={status.db_path}")
    typer.echo(f"schema_version={status.schema_version}")
    typer.echo(f"latest_schema_version={status.latest_schema_version}")
    typer.echo(f"journal_mode={status.journal_mode}")
    typer.echo(f"wal_active={str(status.wal_active).lower()}")
    typer.echo(f"foreign_keys={str(status.foreign_keys).lower()}")
    typer.echo(f"busy_timeout_ms={status.busy_timeout_ms}")


@search_index_app.command("status")
def search_index_status_cmd(
    project: str | None = typer.Option(None, help="Optional project filter (informational)"),
) -> None:
    """Show lexical candidate index completeness (read-only)."""
    from .search_index import search_index_status

    mesh = _mesh()
    with mesh.store.connection() as conn:
        status = search_index_status(conn)
    typer.echo(f"messages={status.messages}")
    typer.echo(f"indexed_messages={status.indexed_messages}")
    typer.echo(f"term_rows={status.term_rows}")
    typer.echo(f"missing_messages={status.missing_messages}")
    typer.echo(f"complete={str(status.complete).lower()}")
    if project:
        typer.echo(f"project_filter={project}")


@search_index_app.command("rebuild")
def search_index_rebuild(
    project: str | None = typer.Option(
        None,
        help="Optional project namespace (rebuild is global; project is logged only)",
    ),
) -> None:
    """Rebuild the lexical candidate term index."""
    from .search_index import rebuild_all_message_terms

    mesh = _mesh()
    mesh.init()

    def _do() -> int:
        with mesh.store.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                count = rebuild_all_message_terms(conn)
                conn.commit()
                return count
            except Exception:
                conn.rollback()
                raise

    from .storage import with_busy_retry

    count = with_busy_retry(_do)
    typer.echo(f"rebuilt_messages={count}")
    if project:
        typer.echo(f"project={project}")


@pull_app.command("api")
def pull_api_cmd(
    url: str = typer.Option(..., help="Source API URL (https:// only unless --allow-private)"),
    project: str = typer.Option(..., help="Project namespace"),
    response_format: str = typer.Option(
        ..., "--format", help="Response body shape: json | jsonl | bundle"
    ),
    provider: str = typer.Option(
        "jsonl",
        help="Provider label attached to ingested rows (parsing is always generic)",
    ),
    token_env: str | None = typer.Option(
        None,
        "--token-env",
        help="Name of an environment variable holding the bearer token (never the token itself)",
    ),
    token_file: Path | None = typer.Option(
        None,
        "--token-file",
        exists=True,
        dir_okay=False,
        help="Path to a file containing the bearer token",
    ),
    timeout: float = typer.Option(30.0, help="Request timeout in seconds"),
    max_bytes: int = typer.Option(
        10 * 1024 * 1024, "--max-bytes", help="Maximum response bytes to read"
    ),
    allow_host: list[str] = typer.Option(
        [],
        "--allow-host",
        help="Allowed destination hostname, exact match (repeatable)",
    ),
    allow_private: bool = typer.Option(
        False,
        "--allow-private",
        help="Explicitly allow http:// and private/loopback destinations (local testing only)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate and fetch without writing to the database"
    ),
) -> None:
    """Pull messages from an explicit, token-gated JSON/JSONL/bundle API endpoint.

    Never claims vendor chat-history semantics (OpenAI/ChatGPT/Claude, etc.) —
    only generic json/jsonl/bundle response shapes are understood. The bearer
    token is read from --token-env or --token-file; it is never accepted as a
    plain CLI value and is never stored in the database, reports, or logs.
    """
    from .api_pull import ApiPullError, pull_api

    mesh = _mesh()
    mesh.init()
    try:
        report = pull_api(
            store=mesh.store,
            settings=mesh.settings,
            project=project,
            url=url,
            fmt=response_format,
            provider=provider,
            token_env=token_env,
            token_file=token_file,
            timeout=timeout,
            max_bytes=max_bytes,
            allow_hosts=allow_host,
            allow_private=allow_private,
            dry_run=dry_run,
        )
    except ApiPullError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"url={report.url}")
    typer.echo(f"status={report.status} conditional={report.conditional}")
    typer.echo(
        f"bytes={report.bytes_received} seen={report.seen} inserted={report.inserted} "
        f"duplicates={report.duplicates} failures={report.failures}"
    )
    if report.dry_run:
        typer.echo("dry_run=yes; no changes were made")
    if report.failure_details:
        for detail in report.failure_details[:20]:
            typer.echo(f"  {detail}", err=True)


@auth_app.command("status")
def auth_status_cmd() -> None:
    """Show HTTP auth readiness and token counts (no secrets, ever)."""
    from .auth import auth_status

    mesh = _mesh()
    status = auth_status(mesh.settings.db_path)
    typer.echo(f"http_auth_mode={mesh.settings.http_auth_mode}")
    typer.echo(f"schema_ready={str(status.schema_ready).lower()}")
    typer.echo(f"total_tokens={status.total_tokens}")
    typer.echo(f"active_tokens={status.active_tokens}")
    typer.echo(f"revoked_tokens={status.revoked_tokens}")
    if status.projects:
        typer.echo(f"projects={','.join(status.projects)}")


@auth_token_app.command("create")
def auth_token_create(
    project: str = typer.Option(..., help="Project namespace this token authorizes"),
    scope: list[str] = typer.Option(
        [],
        "--scope",
        help="Repeatable: read and/or write (at least one required)",
    ),
    label: str | None = typer.Option(None, help="Optional human-readable label"),
    expires: str | None = typer.Option(
        None, "--expires", help="Optional ISO8601 expiry, e.g. 2027-01-01T00:00:00+00:00"
    ),
    token_file: Path | None = typer.Option(
        None,
        "--token-file",
        help="Also write the token to this file (mode 0600) instead of only printing it",
    ),
) -> None:
    """Create a project-scoped bearer token. The full token is shown ONCE."""
    from .auth import AuthError, create_token, write_token_file

    if not scope:
        typer.echo("error: at least one --scope (read and/or write) is required", err=True)
        raise typer.Exit(code=1)

    mesh = _mesh()
    mesh.init()
    try:
        token_plaintext, record = create_token(
            mesh.settings.db_path, project, scope, label=label, expires_at=expires
        )
    except (ValueError, AuthError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Token created. This is the ONLY time the full token is shown — store it now:")
    typer.echo(token_plaintext)
    typer.echo(
        f"token_id={record.token_id} project={record.project} scopes={','.join(record.scopes)}"
    )
    if record.expires_at:
        typer.echo(f"expires_at={record.expires_at}")
    if token_file is not None:
        try:
            write_token_file(token_file, token_plaintext)
        except AuthError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"Wrote token to {token_file} (mode 0600)")


@auth_token_app.command("list")
def auth_token_list(project: str = typer.Option(..., help="Project namespace")) -> None:
    """List tokens for a project. Never prints secrets."""
    from .auth import list_tokens

    mesh = _mesh()
    records = list_tokens(mesh.settings.db_path, project)
    if not records:
        typer.echo("No tokens.")
        return
    for r in records:
        state = "revoked" if r.revoked_at else "active"
        typer.echo(
            f"token_id={r.token_id} project={r.project} scopes={','.join(r.scopes)} "
            f"state={state} created_at={r.created_at} expires_at={r.expires_at or ''} "
            f"last_used_at={r.last_used_at or ''} label={r.label or ''}"
        )


@auth_token_app.command("revoke")
def auth_token_revoke(id: str = typer.Option(..., "--id", help="Token id to revoke")) -> None:
    """Revoke a token by id (idempotent)."""
    from .auth import TokenNotFoundError, revoke_token

    mesh = _mesh()
    try:
        record = revoke_token(mesh.settings.db_path, id)
    except TokenNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"revoked token_id={record.token_id} project={record.project}")


@auth_token_app.command("rotate")
def auth_token_rotate(
    id: str = typer.Option(..., "--id", help="Token id to rotate"),
    token_file: Path | None = typer.Option(
        None,
        "--token-file",
        help="Also write the new token to this file (mode 0600, overwritten if present)",
    ),
) -> None:
    """Revoke a token and issue a new one with the same project/scopes/label."""
    from .auth import TokenNotFoundError, rotate_token, write_token_file

    mesh = _mesh()
    try:
        token_plaintext, record = rotate_token(mesh.settings.db_path, id)
    except TokenNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Token rotated. This is the ONLY time the new full token is shown — store it now:")
    typer.echo(token_plaintext)
    typer.echo(
        f"token_id={record.token_id} project={record.project} scopes={','.join(record.scopes)}"
    )
    if token_file is not None:
        write_token_file(token_file, token_plaintext, overwrite=True)
        typer.echo(f"Wrote token to {token_file} (mode 0600)")


@encryption_app.command("status")
def encryption_status_cmd() -> None:
    """Show encryption readiness (never prints key material)."""
    from .encryption import encryption_status

    mesh = _mesh()
    status = encryption_status(mesh.settings.db_path)
    typer.echo(f"schema_ready={str(status.schema_ready).lower()}")
    typer.echo(f"enabled={str(status.enabled).lower()}")
    typer.echo(f"key_id={status.key_id or ''}")
    typer.echo(f"algorithm={status.algorithm or ''}")
    typer.echo(f"envelope_version={status.envelope_version if status.envelope_version is not None else ''}")
    typer.echo(f"terms_mode={status.terms_mode}")
    typer.echo(f"cryptography_available={str(status.cryptography_available).lower()}")
    typer.echo(
        "threat_model=protects stolen/copied DB+backups without the key; "
        "does NOT protect same-user malware, key theft, or full workstation compromise. "
        "Schema/ids/timestamps remain visible metadata."
    )


@encryption_app.command("key-generate")
def encryption_key_generate(
    key_file: Path = typer.Option(..., "--key-file", help="Destination path for a new key file"),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace an existing key file (dangerous if still in use)",
    ),
) -> None:
    """Generate a 256-bit key file (mode 0600). Does not print the key."""
    from .crypto import KeyFileExistsError, generate_key_file

    try:
        path = generate_key_file(key_file, overwrite=overwrite)
    except KeyFileExistsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote key file to {path} (mode 0600). Store it separately from backups.")
    typer.echo(
        "A key beside the database protects copied DBs without the key, "
        "but does not protect full-home compromise."
    )


@encryption_app.command("enable")
def encryption_enable(
    key_file: Path = typer.Option(..., "--key-file", exists=True, dir_okay=False),
) -> None:
    """Enable field-level encryption. Creates a plaintext-sensitive pre-encryption backup."""
    from .crypto import InvalidKeyMaterialError, load_key_from_file
    from .encryption import EncryptionLifecycleError, enable_encryption

    mesh = _mesh()
    mesh.init()
    try:
        master = load_key_from_file(key_file)
        report = enable_encryption(mesh.settings.db_path, master_key=master)
    except (EncryptionLifecycleError, InvalidKeyMaterialError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    mesh.store._invalidate_crypto_cache()
    mesh.settings.encryption_key_file = key_file
    mesh.settings.save()
    typer.echo(f"encryption enabled key_id={report.key_id} terms_mode={report.terms_mode}")
    typer.echo(
        f"encrypted messages={report.messages_encrypted} "
        f"embeddings={report.embeddings_encrypted} "
        f"summaries={report.summaries_encrypted} "
        f"agent_state={report.agent_state_encrypted}"
    )
    if report.backup_path is not None:
        typer.echo(
            f"WARNING: pre-encryption backup contains PLAINTEXT: {report.backup_path}",
            err=True,
        )
        typer.echo("The plaintext backup is NOT deleted automatically.", err=True)
    typer.echo(f"vacuumed={str(report.vacuumed).lower()} verified_samples={report.verified_samples}")


@encryption_app.command("rotate")
def encryption_rotate(
    key_file: Path = typer.Option(
        ...,
        "--key-file",
        exists=True,
        dir_okay=False,
        help="Current key file",
    ),
    new_key_file: Path = typer.Option(
        ...,
        "--new-key-file",
        exists=True,
        dir_okay=False,
        help="New key file",
    ),
) -> None:
    """Re-encrypt all protected fields under a new key. Wrong old key makes no changes."""
    from .crypto import InvalidKeyMaterialError, load_key_from_file
    from .encryption import EncryptionLifecycleError, rotate_encryption

    mesh = _mesh()
    mesh.init()
    try:
        old_key = load_key_from_file(key_file)
        new_key = load_key_from_file(new_key_file)
        report = rotate_encryption(
            mesh.settings.db_path,
            old_key=old_key,
            new_key=new_key,
        )
    except (EncryptionLifecycleError, InvalidKeyMaterialError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    mesh.store._invalidate_crypto_cache()
    mesh.settings.encryption_key_file = new_key_file
    mesh.settings.save()
    typer.echo(
        f"encryption rotated previous_key_id={report.previous_key_id} "
        f"key_id={report.key_id}"
    )
    if report.backup_path is not None:
        typer.echo(f"pre-rotate backup: {report.backup_path}")


if __name__ == "__main__":
    app()
