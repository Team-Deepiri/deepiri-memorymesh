"""Transactional native integration installation and uninstall (T12/T36).

Installs are preflighted, backed up, staged, atomically replaced, and rolled
back on failure. Manifests under ``~/.config/deepiri-memorymesh/integrations/``
never store bearer tokens or encryption keys — only token-*file* path
references and non-secret ownership markers.

Failure injection for tests: set ``_INJECT_FAILURE_AT`` to a phase name
(``after_plan``, ``after_backup``, ``after_stage``, ``after_replace``,
``after_verify``, ``before_manifest``, ``uninstall_after_config_stage``,
``uninstall_after_quarantine``, ``uninstall_before_manifest_delete``).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .integrations import (
    OWNERSHIP_MARKER_KEY,
    _append_owned_command_hook,
    _append_owned_stop_hook,
    _assert_integration_allowed,
    _load_json,
    render_aider_wrapper,
    render_bridge_script,
    render_hook_script,
    render_opencode_plugin,
)
from .providers.registry import get_provider

MANIFEST_FORMAT_VERSION = 1

# Test-only failure injection point. Production code leaves this None.
_INJECT_FAILURE_AT: str | None = None


def _maybe_inject(phase: str) -> None:
    if _INJECT_FAILURE_AT is not None and _INJECT_FAILURE_AT == phase:
        raise RuntimeError(f"injected failure at phase={phase}")


def integrations_manifest_dir() -> Path:
    return Path.home() / ".config" / "deepiri-memorymesh" / "integrations"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_text(path: Path, text: str, *, mode: int = 0o644) -> None:
    """Stage on the same filesystem, fsync, then atomically replace *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _ownership_marker(installation_id: str, provider: str) -> dict[str, str]:
    return {
        "product": "memorymesh",
        "installation_id": installation_id,
        "provider": provider,
    }


@dataclass(slots=True)
class GeneratedFileSpec:
    path: str
    sha256: str
    kind: str  # bridge|hook|plugin|wrapper|config


@dataclass(slots=True)
class ConfigEditSpec:
    path: str
    backup_path: str
    marker: dict[str, str]
    event: str | None = None


@dataclass(slots=True)
class InstallManifest:
    format_version: int
    installation_id: str
    provider: str
    project: str
    service_url: str
    token_file: str | None
    generated_files: list[GeneratedFileSpec] = field(default_factory=list)
    config_edits: list[ConfigEditSpec] = field(default_factory=list)
    installed_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "installation_id": self.installation_id,
            "provider": self.provider,
            "project": self.project,
            "service_url": self.service_url,
            "token_file": self.token_file,
            "generated_files": [asdict(g) for g in self.generated_files],
            "config_edits": [asdict(c) for c in self.config_edits],
            "installed_at": self.installed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InstallManifest":
        return cls(
            format_version=int(data.get("format_version") or MANIFEST_FORMAT_VERSION),
            installation_id=str(data["installation_id"]),
            provider=str(data["provider"]),
            project=str(data["project"]),
            service_url=str(data["service_url"]),
            token_file=data.get("token_file"),
            generated_files=[
                GeneratedFileSpec(**g) for g in (data.get("generated_files") or [])
            ],
            config_edits=[ConfigEditSpec(**c) for c in (data.get("config_edits") or [])],
            installed_at=str(data.get("installed_at") or ""),
        )


@dataclass(slots=True)
class InstallPlan:
    provider: str
    project: str
    service_url: str
    token_file: str | None
    actions: list[str] = field(default_factory=list)
    would_write: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InstallReport:
    ok: bool
    dry_run: bool = False
    noop: bool = False
    installation_id: str | None = None
    manifest_path: Path | None = None
    plan: InstallPlan | None = None
    message: str = ""
    rolled_back: bool = False
    rollback_ok: bool = True
    rollback_errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class UninstallReport:
    ok: bool
    dry_run: bool = False
    noop: bool = False
    installation_id: str | None = None
    removed_files: list[str] = field(default_factory=list)
    preserved_files: list[str] = field(default_factory=list)
    config_changes: list[str] = field(default_factory=list)
    message: str = ""
    conflicts: list[str] = field(default_factory=list)
    rolled_back: bool = False
    rollback_ok: bool = True
    rollback_errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FileWritePlan:
    """One destination that will be written under transactional control."""

    dest: Path
    content: str
    mode: int
    kind: str  # generated | config
    role: str  # bridge|hook|plugin|wrapper|config
    event: str | None = None
    preexisting_bytes: bytes | None = None
    backup_path: Path | None = None
    was_created: bool = False


def _manifest_path(installation_id: str) -> Path:
    return integrations_manifest_dir() / f"{installation_id}.json"


def _find_manifest_for_provider(provider: str) -> InstallManifest | None:
    root = integrations_manifest_dir()
    if not root.exists():
        return None
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("provider") == provider:
            return InstallManifest.from_dict(data)
    return None


def list_installations() -> list[InstallManifest]:
    root = integrations_manifest_dir()
    if not root.exists():
        return []
    out: list[InstallManifest] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "installation_id" in data:
                out.append(InstallManifest.from_dict(data))
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def verify_installation(installation_id: str) -> dict[str, Any]:
    path = _manifest_path(installation_id)
    if not path.exists():
        return {"ok": False, "error": "manifest_not_found"}
    manifest = InstallManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    missing: list[str] = []
    hash_mismatch: list[str] = []
    for gen in manifest.generated_files:
        p = Path(gen.path)
        if not p.exists():
            missing.append(gen.path)
            continue
        if _sha256_file(p) != gen.sha256:
            hash_mismatch.append(gen.path)
    return {
        "ok": not missing and not hash_mismatch,
        "installation_id": installation_id,
        "missing": missing,
        "hash_mismatch": hash_mismatch,
        "provider": manifest.provider,
        "project": manifest.project,
        "token_file": manifest.token_file,
    }


def _validate_token_source(token_file: Path | None, *, auth_required: bool) -> None:
    if not auth_required:
        return
    if token_file is None:
        raise ValueError(
            "HTTP auth is required; pass --token-file referencing a project token "
            "(token content is never stored in the manifest)."
        )
    if not token_file.exists() or not token_file.is_file():
        raise ValueError(f"token-file does not exist: {token_file}")
    try:
        mode = token_file.stat().st_mode & 0o777
        if mode & 0o077:
            pass
    except OSError:
        pass


def _load_and_validate_json_config(path: Path) -> dict:
    """Parse existing JSON config or return {}. Reject non-object roots."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed configuration at {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"configuration at {path} must be a JSON object")
    return raw


def _backup_path_for(dest: Path, installation_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return dest.with_name(f"{dest.name}.pre-install.{installation_id}.{stamp}.bak")


def _build_provider_writes(
    *,
    key: str,
    installation_id: str,
    project: str,
    service_url: str,
) -> tuple[InstallPlan, list[FileWritePlan]]:
    """Pure planning: compute every file content without mutating the filesystem.

    May *read* existing configs for validation/merge. Never writes.
    """
    marker = _ownership_marker(installation_id, key)
    writes: list[FileWritePlan] = []
    actions: list[str] = []
    would_write: list[str] = []

    if key == "opencode":
        plugin_path, body = render_opencode_plugin(
            project, service_url, installation_id=installation_id
        )
        preexisting = plugin_path.read_bytes() if plugin_path.exists() else None
        writes.append(
            FileWritePlan(
                dest=plugin_path,
                content=body,
                mode=0o644,
                kind="generated",
                role="plugin",
                preexisting_bytes=preexisting,
            )
        )
        actions.append("write_opencode_plugin")
        would_write.append(str(plugin_path))
    elif key == "aider":
        # Aider uses bridge/hook targets labeled "jsonl" for ingest provider key.
        bridge_path, bridge_body = render_bridge_script(
            "jsonl", project, service_url, installation_id=installation_id
        )
        hook_path, hook_body = render_hook_script(
            "jsonl", project, service_url, installation_id=installation_id
        )
        wrapper_path, wrapper_body = render_aider_wrapper(
            project, installation_id=installation_id
        )
        for dest, body, role, mode in (
            (bridge_path, bridge_body, "bridge", 0o755),
            (hook_path, hook_body, "hook", 0o755),
            (wrapper_path, wrapper_body, "wrapper", 0o755),
        ):
            preexisting = dest.read_bytes() if dest.exists() else None
            writes.append(
                FileWritePlan(
                    dest=dest,
                    content=body,
                    mode=mode,
                    kind="generated",
                    role=role,
                    preexisting_bytes=preexisting,
                )
            )
            would_write.append(str(dest))
        actions.extend(["write_bridge", "write_hook", "write_aider_wrapper"])
    else:
        # claude / cursor / gemini / continue: bridge + hook + owned config edit
        bridge_path, bridge_body = render_bridge_script(
            key, project, service_url, installation_id=installation_id
        )
        hook_path, hook_body = render_hook_script(
            key, project, service_url, installation_id=installation_id
        )
        for dest, body, role in (
            (bridge_path, bridge_body, "bridge"),
            (hook_path, hook_body, "hook"),
        ):
            preexisting = dest.read_bytes() if dest.exists() else None
            writes.append(
                FileWritePlan(
                    dest=dest,
                    content=body,
                    mode=0o755,
                    kind="generated",
                    role=role,
                    preexisting_bytes=preexisting,
                )
            )
            would_write.append(str(dest))
        actions.extend(["write_bridge", "write_hook"])

        if key == "claude":
            cfg_path = Path.home() / ".claude" / "settings.json"
            event = "SessionEnd"
            cfg = _load_and_validate_json_config(cfg_path)
            cfg = _append_owned_command_hook(cfg, event, str(hook_path), marker)
            content = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
            preexisting = cfg_path.read_bytes() if cfg_path.exists() else None
            writes.append(
                FileWritePlan(
                    dest=cfg_path,
                    content=content,
                    mode=0o644,
                    kind="config",
                    role="config",
                    event=event,
                    preexisting_bytes=preexisting,
                )
            )
            actions.append("edit_claude_settings")
            would_write.append(str(cfg_path))
        elif key == "cursor":
            cfg_path = Path.home() / ".cursor" / "hooks.json"
            cfg = _load_and_validate_json_config(cfg_path)
            cfg = _append_owned_stop_hook(cfg, str(hook_path), marker)
            content = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
            preexisting = cfg_path.read_bytes() if cfg_path.exists() else None
            writes.append(
                FileWritePlan(
                    dest=cfg_path,
                    content=content,
                    mode=0o644,
                    kind="config",
                    role="config",
                    event="stop",
                    preexisting_bytes=preexisting,
                )
            )
            actions.append("edit_cursor_hooks")
            would_write.append(str(cfg_path))
        elif key == "gemini":
            cfg_path = Path.home() / ".gemini" / "settings.json"
            event = "SessionEnd"
            cfg = _load_and_validate_json_config(cfg_path)
            cfg = _append_owned_command_hook(cfg, event, str(hook_path), marker)
            content = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
            preexisting = cfg_path.read_bytes() if cfg_path.exists() else None
            writes.append(
                FileWritePlan(
                    dest=cfg_path,
                    content=content,
                    mode=0o644,
                    kind="config",
                    role="config",
                    event=event,
                    preexisting_bytes=preexisting,
                )
            )
            actions.append("edit_gemini_settings")
            would_write.append(str(cfg_path))
        elif key == "continue":
            cfg_path = Path.home() / ".continue" / "settings.json"
            event = "SessionEnd"
            cfg = _load_and_validate_json_config(cfg_path)
            cfg = _append_owned_command_hook(cfg, event, str(hook_path), marker)
            content = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
            preexisting = cfg_path.read_bytes() if cfg_path.exists() else None
            writes.append(
                FileWritePlan(
                    dest=cfg_path,
                    content=content,
                    mode=0o644,
                    kind="config",
                    role="config",
                    event=event,
                    preexisting_bytes=preexisting,
                )
            )
            actions.append("edit_continue_settings")
            would_write.append(str(cfg_path))
        else:
            raise ValueError(f"unsupported native provider for transactional install: {key!r}")

    plan = InstallPlan(
        provider=key,
        project=project,
        service_url=service_url,
        token_file=None,  # filled by caller
        actions=actions,
        would_write=would_write,
    )
    return plan, writes


def _rollback_install(
    writes: list[FileWritePlan],
    *,
    installation_id: str,
    replaced: list[FileWritePlan],
) -> list[str]:
    """Restore backups / remove newly created files / drop incomplete manifest."""
    errors: list[str] = []
    for item in reversed(replaced):
        try:
            if item.preexisting_bytes is not None and item.backup_path is not None:
                if item.backup_path.exists():
                    shutil.copy2(item.backup_path, item.dest)
                else:
                    item.dest.write_bytes(item.preexisting_bytes)
            elif item.was_created or item.preexisting_bytes is None:
                item.dest.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"restore {item.dest}: {exc}")
    try:
        _manifest_path(installation_id).unlink(missing_ok=True)
    except OSError as exc:
        errors.append(f"manifest: {exc}")
    # Clean leftover .bak from this install id under each dest parent (best effort).
    for item in writes:
        if item.backup_path is not None:
            try:
                item.backup_path.unlink(missing_ok=True)
            except OSError as exc:
                errors.append(f"backup cleanup {item.backup_path}: {exc}")
    return errors


def install_native_transactional(
    target: str,
    project: str,
    service_url: str = "http://127.0.0.1:8765",
    *,
    token_file: Path | None = None,
    dry_run: bool = False,
    update: bool = False,
    auth_required: bool = True,
) -> InstallReport:
    """Install one native integration fully transactionally (T12).

    Never calls mutating helpers that write outside this transaction manager.
    """
    key = _assert_integration_allowed(target)
    if get_provider(key).parser_kind == "generic-explicit":
        raise ValueError(
            f"Provider {target!r} is generic-explicit; use install-integration/bridge, "
            "not install-native"
        )

    _validate_token_source(token_file, auth_required=auth_required)
    token_ref = str(token_file) if token_file is not None else None

    existing = _find_manifest_for_provider(key)
    if existing is not None:
        same = (
            existing.project == project
            and existing.service_url == service_url
            and (existing.token_file or None) == token_ref
        )
        if same and not update:
            return InstallReport(
                ok=True,
                noop=True,
                installation_id=existing.installation_id,
                message="identical installation already present; no-op",
            )
        if not same and not update:
            return InstallReport(
                ok=False,
                installation_id=existing.installation_id,
                message=(
                    f"existing installation for {key} differs "
                    f"(project/service_url/token_file). Pass --update to replace."
                ),
                plan=InstallPlan(
                    provider=key,
                    project=project,
                    service_url=service_url,
                    token_file=token_ref,
                    conflicts=[
                        f"existing installation_id={existing.installation_id}",
                    ],
                ),
            )
        # update=True: uninstall prior installation first (transactional), then install.
        if update and existing is not None:
            un = uninstall_native_transactional(
                key, force=True, installation_id=existing.installation_id
            )
            if not un.ok and not un.noop:
                return InstallReport(
                    ok=False,
                    installation_id=existing.installation_id,
                    message=f"failed to replace existing installation: {un.message}",
                    rollback_errors=list(un.conflicts),
                )

    installation_id = secrets.token_hex(8)

    try:
        plan, writes = _build_provider_writes(
            key=key,
            installation_id=installation_id,
            project=project,
            service_url=service_url,
        )
    except ValueError as exc:
        return InstallReport(ok=False, message=str(exc))

    plan.token_file = token_ref
    _maybe_inject("after_plan")

    if dry_run:
        return InstallReport(ok=True, dry_run=True, plan=plan, message="dry-run")

    replaced: list[FileWritePlan] = []
    try:
        # Phase: backups (non-overwriting)
        for item in writes:
            if item.preexisting_bytes is not None:
                bak = _backup_path_for(item.dest, installation_id)
                if bak.exists():
                    raise RuntimeError(f"backup path exists: {bak}")
                item.dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item.dest, bak)
                item.backup_path = bak
        _maybe_inject("after_backup")

        # Phase: stage + atomic replace (same FS via mkstemp in dest parent)
        for item in writes:
            created = not item.dest.exists()
            _atomic_write_text(item.dest, item.content, mode=item.mode)
            item.was_created = created
            replaced.append(item)
        _maybe_inject("after_stage")
        _maybe_inject("after_replace")

        # Phase: verify hashes and ownership markers
        for item in writes:
            if _sha256_file(item.dest) != _sha256_text(item.content):
                raise RuntimeError(f"post-install hash mismatch: {item.dest}")
            if item.kind == "generated" and item.role != "plugin":
                text = item.dest.read_text(encoding="utf-8")
                if f"installation_id={installation_id}" not in text:
                    raise RuntimeError(f"missing ownership marker: {item.dest}")
            if item.kind == "generated" and item.role == "plugin":
                text = item.dest.read_text(encoding="utf-8")
                if f"installation_id={installation_id}" not in text:
                    raise RuntimeError(f"missing ownership marker: {item.dest}")
            if item.kind == "config":
                cfg = _load_json(item.dest)
                blob = json.dumps(cfg, ensure_ascii=True)
                if installation_id not in blob:
                    raise RuntimeError(f"owned config entry missing: {item.dest}")
        _maybe_inject("after_verify")

        # Phase: manifest last
        generated = [
            GeneratedFileSpec(
                path=str(item.dest),
                sha256=_sha256_text(item.content),
                kind=item.role,
            )
            for item in writes
            if item.kind == "generated"
        ]
        config_edits: list[ConfigEditSpec] = []
        for item in writes:
            if item.kind != "config":
                continue
            config_edits.append(
                ConfigEditSpec(
                    path=str(item.dest),
                    backup_path=str(item.backup_path) if item.backup_path else "",
                    marker=_ownership_marker(installation_id, key),
                    event=item.event,
                )
            )
        # Also record preexisting generated backups so restore_from_backup works.
        for item in writes:
            if item.kind == "generated" and item.backup_path is not None:
                config_edits.append(
                    ConfigEditSpec(
                        path=str(item.dest),
                        backup_path=str(item.backup_path),
                        marker=_ownership_marker(installation_id, key),
                        event=None,
                    )
                )

        manifest = InstallManifest(
            format_version=MANIFEST_FORMAT_VERSION,
            installation_id=installation_id,
            provider=key,
            project=project,
            service_url=service_url,
            token_file=token_ref,
            generated_files=generated,
            config_edits=config_edits,
        )
        _maybe_inject("before_manifest")
        mpath = _manifest_path(installation_id)
        _atomic_write_text(
            mpath,
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=True) + "\n",
            mode=0o600,
        )
        return InstallReport(
            ok=True,
            installation_id=installation_id,
            manifest_path=mpath,
            plan=plan,
            message="ok",
        )
    except Exception as exc:
        rollback_errors = _rollback_install(
            writes, installation_id=installation_id, replaced=replaced
        )
        return InstallReport(
            ok=False,
            installation_id=installation_id,
            plan=plan,
            rolled_back=True,
            rollback_ok=not rollback_errors,
            rollback_errors=rollback_errors,
            message=(
                f"install failed and "
                f"{'rolled back' if not rollback_errors else 'rollback incomplete'}: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


def _remove_owned_hook_entries(config: dict, marker: dict[str, str]) -> dict:
    """Remove hook entries that carry our ownership marker or command path marker."""
    hooks = config.get("hooks")
    if not isinstance(hooks, dict):
        return config
    installation_id = marker.get("installation_id", "")
    for event, entries in list(hooks.items()):
        if not isinstance(entries, list):
            continue
        kept = []
        for entry in entries:
            if not isinstance(entry, dict):
                kept.append(entry)
                continue
            owned = entry.get(OWNERSHIP_MARKER_KEY)
            if (
                isinstance(owned, dict)
                and owned.get("installation_id") == installation_id
            ):
                continue
            cmd_blob = json.dumps(entry, ensure_ascii=True)
            if installation_id and installation_id in cmd_blob:
                continue
            kept.append(entry)
        hooks[event] = kept
    return config


def uninstall_native_transactional(
    target: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    installation_id: str | None = None,
) -> UninstallReport:
    """Remove only Memory Mesh-owned artifacts for a target (T36 transactional)."""
    key = _assert_integration_allowed(target)
    manifest = None
    if installation_id:
        path = _manifest_path(installation_id)
        if path.exists():
            manifest = InstallManifest.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
    else:
        manifest = _find_manifest_for_provider(key)

    if manifest is None:
        return UninstallReport(
            ok=True,
            noop=True,
            message=f"no installation found for {key}",
        )

    if dry_run:
        return UninstallReport(
            ok=True,
            dry_run=True,
            installation_id=manifest.installation_id,
            removed_files=[g.path for g in manifest.generated_files],
            message="dry-run",
        )

    removed: list[str] = []
    preserved: list[str] = []
    conflicts: list[str] = []
    config_changes: list[str] = []
    quarantine_dir = (
        integrations_manifest_dir()
        / ".quarantine"
        / manifest.installation_id
    )
    staged_configs: list[tuple[Path, Path, bytes]] = []  # (dest, staged, original)
    quarantined: list[tuple[Path, Path]] = []  # (original, quarantine)

    try:
        # Plan config mutations first (pure).
        planned_cfg: list[tuple[Path, str, bytes]] = []
        for edit in manifest.config_edits:
            cfg_path = Path(edit.path)
            if not cfg_path.exists():
                continue
            if cfg_path.suffix in {".ts", ".js", ".mjs"}:
                continue
            # Skip generated backup records accidentally filed as config_edits.
            if any(g.path == edit.path for g in manifest.generated_files):
                # Still may be a real config that equals a generated path — rare.
                # Only skip non-JSON.
                if cfg_path.suffix not in {".json"}:
                    continue
            try:
                original = cfg_path.read_bytes()
                cfg = json.loads(original.decode("utf-8"))
                if not isinstance(cfg, dict):
                    conflicts.append(f"malformed config not touched: {cfg_path}")
                    continue
                before = json.dumps(cfg, sort_keys=True)
                cfg = _remove_owned_hook_entries(cfg, edit.marker)
                after = json.dumps(cfg, sort_keys=True)
                if before != after:
                    new_text = json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
                    planned_cfg.append((cfg_path, new_text, original))
            except Exception as exc:
                conflicts.append(f"config plan failed for {cfg_path}: {exc}")

        if any(c.startswith("config plan failed") for c in conflicts):
            raise RuntimeError("; ".join(conflicts))

        # Stage + replace configs
        for cfg_path, new_text, original in planned_cfg:
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{cfg_path.name}.", dir=str(cfg_path.parent)
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(new_text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, cfg_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
            staged_configs.append((cfg_path, tmp_path, original))
            config_changes.append(str(cfg_path))
        _maybe_inject("uninstall_after_config_stage")

        # Quarantine generated files (after hash/ownership checks)
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for gen in manifest.generated_files:
            p = Path(gen.path)
            if not p.exists():
                removed.append(f"missing:{gen.path}")
                continue
            current = _sha256_file(p)
            if current != gen.sha256 and not force:
                preserved.append(gen.path)
                conflicts.append(f"modified generated file preserved: {gen.path}")
                continue
            q_dest = quarantine_dir / f"{p.name}.{secrets.token_hex(4)}"
            shutil.move(str(p), str(q_dest))
            quarantined.append((p, q_dest))
            removed.append(gen.path)
        _maybe_inject("uninstall_after_quarantine")

        # Manifest only after all owned changes succeed (unless preserved files).
        if preserved:
            conflicts.append(
                "manifest retained because modified generated files were preserved; "
                "re-run with --force to remove them"
            )
            # Restore nothing — configs already cleaned of owned entries is OK;
            # still leave manifest so --force can finish.
            # Delete quarantine now for successfully moved files? Keep until force.
            # Actually leave quarantined files deleted after success path;
            # for preserved case we already didn't quarantine those.
            for _orig, q in quarantined:
                try:
                    q.unlink(missing_ok=True)
                except OSError as exc:
                    conflicts.append(f"quarantine cleanup: {exc}")
            try:
                if quarantine_dir.exists() and not any(quarantine_dir.iterdir()):
                    quarantine_dir.rmdir()
            except OSError:
                pass
            return UninstallReport(
                ok=True,
                installation_id=manifest.installation_id,
                removed_files=removed,
                preserved_files=preserved,
                config_changes=config_changes,
                conflicts=conflicts,
                message="completed with preserved files",
            )

        _maybe_inject("uninstall_before_manifest_delete")
        try:
            _manifest_path(manifest.installation_id).unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(f"manifest remove failed: {exc}") from exc

        # Final delete quarantined files
        for _orig, q in quarantined:
            try:
                q.unlink(missing_ok=True)
            except OSError as exc:
                conflicts.append(f"quarantine final delete: {exc}")
        try:
            if quarantine_dir.exists() and not any(quarantine_dir.iterdir()):
                quarantine_dir.rmdir()
        except OSError:
            pass

        return UninstallReport(
            ok=not any(c.startswith("quarantine final") for c in conflicts),
            installation_id=manifest.installation_id,
            removed_files=removed,
            preserved_files=preserved,
            config_changes=config_changes,
            conflicts=conflicts,
            message="ok",
        )
    except Exception as exc:
        rollback_errors: list[str] = []
        # Restore configs from originals
        for cfg_path, _tmp, original in reversed(staged_configs):
            try:
                cfg_path.write_bytes(original)
            except OSError as rb_exc:
                rollback_errors.append(f"config restore {cfg_path}: {rb_exc}")
        # Restore quarantined files
        for orig, q in reversed(quarantined):
            try:
                if q.exists():
                    shutil.move(str(q), str(orig))
            except OSError as rb_exc:
                rollback_errors.append(f"quarantine restore {orig}: {rb_exc}")
        # Retain manifest
        return UninstallReport(
            ok=False,
            installation_id=manifest.installation_id,
            removed_files=removed,
            preserved_files=preserved,
            config_changes=[],
            conflicts=conflicts,
            rolled_back=True,
            rollback_ok=not rollback_errors,
            rollback_errors=rollback_errors,
            message=(
                f"uninstall failed and "
                f"{'rolled back' if not rollback_errors else 'rollback incomplete'}: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


def uninstall_all_transactional(
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[UninstallReport]:
    """Uninstall every known installation independently (one result each)."""
    results: list[UninstallReport] = []
    for manifest in list_installations():
        results.append(
            uninstall_native_transactional(
                manifest.provider,
                dry_run=dry_run,
                force=force,
                installation_id=manifest.installation_id,
            )
        )
    return results


def restore_from_backup(
    installation_id: str,
    *,
    yes: bool = False,
) -> dict[str, Any]:
    """Restore config/plugin files from install-time backups after confirmation."""
    if not yes:
        return {
            "ok": False,
            "error": "confirmation_required",
            "message": "Pass --yes to restore backups for this installation.",
        }
    path = _manifest_path(installation_id)
    if not path.exists():
        return {"ok": False, "error": "manifest_not_found"}
    manifest = InstallManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    restored: list[str] = []
    for edit in manifest.config_edits:
        if not edit.backup_path:
            continue
        backup = Path(edit.backup_path)
        target = Path(edit.path)
        if not backup.exists():
            continue
        if installation_id not in backup.name:
            return {
                "ok": False,
                "error": "backup_installation_mismatch",
                "message": f"Backup {backup} does not belong to {installation_id}",
            }
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if target.exists():
            safety = target.with_name(f"{target.name}.pre-restore.{stamp}.bak")
            shutil.copy2(target, safety)
        shutil.copy2(backup, target)
        restored.append(str(target))
    return {
        "ok": True,
        "installation_id": installation_id,
        "restored": restored,
    }
