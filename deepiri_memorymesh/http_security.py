"""HTTP service security helpers (loopback bind + ingest path allowlist)."""

from __future__ import annotations

import ipaddress
import os
import socket
from pathlib import Path


MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MiB

DEFAULT_TRANSFER_DIR = Path.home() / ".config" / "deepiri-memorymesh" / "transfers"

_LOOPBACK_HELP = (
    "This MemoryMesh service is local-only; use 127.0.0.1, localhost, or ::1."
)


class HttpApiError(Exception):
    """Controlled HTTP API error with status code and client-safe payload."""

    def __init__(self, status: int, error: str, message: str = "") -> None:
        self.status = status
        self.error = error
        self.message = message or error
        super().__init__(self.message)

    def payload(self) -> dict[str, object]:
        body: dict[str, object] = {"ok": False, "error": self.error}
        if self.message and self.message != self.error:
            body["message"] = self.message
        return body


class IngestPathError(HttpApiError):
    """Raised when a caller-supplied ingest path is rejected."""


def assert_loopback_host(host: str) -> None:
    """
    Require that ``host`` resolves exclusively to loopback addresses.

    Every candidate — including literals and hostnames such as ``localhost`` —
    is resolved with ``socket.getaddrinfo``. Every returned address must be
    loopback. Unspecified binds (``0.0.0.0``, ``::``, ``*``) are rejected.
    """
    raw = (host or "").strip()
    if not raw:
        raise ValueError(f"Host must be a loopback address. {_LOOPBACK_HELP}")

    lowered = raw.lower()
    # Unspecified / all-interfaces binds are never allowed.
    if lowered in {"0.0.0.0", "::", "*"}:
        raise ValueError(f"Refusing to bind to non-loopback host {host!r}. {_LOOPBACK_HELP}")

    try:
        infos = socket.getaddrinfo(raw, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(
            f"Cannot resolve host {host!r} as loopback-only: {exc}. {_LOOPBACK_HELP}"
        ) from exc

    if not infos:
        raise ValueError(f"Cannot resolve host {host!r}. {_LOOPBACK_HELP}")

    for info in infos:
        ip_str = info[4][0]
        # Strip IPv6 zone index if present (e.g. fe80::1%lo0).
        if "%" in ip_str:
            ip_str = ip_str.split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise ValueError(
                f"Refusing to bind to host {host!r} (unrecognized address {ip_str!r}). "
                f"{_LOOPBACK_HELP}"
            ) from exc
        if not ip.is_loopback:
            raise ValueError(
                f"Refusing to bind to non-loopback host {host!r} "
                f"(resolves to {ip_str}). {_LOOPBACK_HELP}"
            )


def bind_address_family(host: str) -> int:
    """
    Choose ``socket.AF_INET`` or ``socket.AF_INET6`` for binding ``host``.

    Prefer IPv4 when the name resolves to an IPv4 loopback address so that
    ``localhost`` continues to use the IPv4 server. Use IPv6 when binding a
    literal IPv6 loopback (``::1``) or when only IPv6 loopback addresses exist.
    """
    raw = (host or "").strip()
    try:
        ip = ipaddress.ip_address(raw)
    except ValueError:
        ip = None

    if ip is not None:
        return socket.AF_INET6 if isinstance(ip, ipaddress.IPv6Address) else socket.AF_INET

    infos = socket.getaddrinfo(raw, None, type=socket.SOCK_STREAM)
    families = {info[0] for info in infos}
    if socket.AF_INET in families:
        return socket.AF_INET
    if socket.AF_INET6 in families:
        return socket.AF_INET6
    return socket.AF_INET


def normalize_ingest_roots(
    provider_paths: dict[str, str] | None = None,
    extra_roots: list[Path | str] | None = None,
    transfer_dir: Path | str | None = None,
) -> list[Path]:
    """
    Build a deduplicated list of canonical allowed ingest root directories.

    Includes configured provider paths, the transfer directory, and any
    explicit extra roots. Does not include the entire home directory.
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    def _add(raw: Path | str) -> None:
        text = str(raw).strip()
        if not text:
            return
        path = Path(text).expanduser().resolve()
        if path in seen:
            return
        seen.add(path)
        roots.append(path)

    if provider_paths:
        for value in provider_paths.values():
            if value:
                _add(value)

    xfer = DEFAULT_TRANSFER_DIR if transfer_dir is None else Path(transfer_dir)
    _add(xfer)

    if extra_roots:
        for root in extra_roots:
            _add(root)

    return roots


def _open_allowlisted_file(base: str, verified_rel: str) -> Path:
    """
    Walk from a trusted ``base`` to ``verified_rel`` using directory listings.

    Each segment is matched against ``os.scandir`` names so filesystem probes
    use OS-returned paths under the allowlisted root, not the raw caller string.
    """
    cur = base
    parts = [p for p in verified_rel.split(os.sep) if p and p != "."]
    if any(p == ".." for p in parts):
        raise IngestPathError(403, "path_not_allowed", "file_path is outside allowed ingest roots")

    for idx, part in enumerate(parts):
        try:
            with os.scandir(cur) as entries:
                match = next((e for e in entries if e.name == part), None)
        except FileNotFoundError as exc:
            raise IngestPathError(404, "path_not_found", "Ingest file not found") from exc
        except OSError as exc:
            raise IngestPathError(400, "invalid_path", "Invalid file_path") from exc

        if match is None:
            raise IngestPathError(404, "path_not_found", "Ingest file not found")

        is_last = idx == len(parts) - 1
        try:
            is_dir = match.is_dir(follow_symlinks=False)
            is_file = match.is_file(follow_symlinks=False)
        except OSError as exc:
            raise IngestPathError(400, "invalid_path", "Invalid file_path") from exc

        if is_last:
            if not is_file:
                raise IngestPathError(400, "invalid_path", "file_path must be a regular file")
            # Re-resolve to reject symlink escapes after the listing match.
            try:
                resolved = os.path.realpath(match.path)
            except OSError as exc:
                raise IngestPathError(400, "invalid_path", "Invalid file_path") from exc
            base_prefix = base if base.endswith(os.sep) else base + os.sep
            if resolved != base and not resolved.startswith(base_prefix):
                raise IngestPathError(
                    403, "path_not_allowed", "file_path is outside allowed ingest roots"
                )
            return Path(resolved)

        if not is_dir:
            raise IngestPathError(400, "invalid_path", "file_path must be a regular file")
        cur = match.path

    # verified_rel empty → the root itself (never a regular file for ingest).
    raise IngestPathError(400, "invalid_path", "file_path must be a regular file")


def validate_ingest_file_path(file_path: str | Path, allowed_roots: list[Path]) -> Path:
    """
    Validate a caller-supplied ingest path against allowed roots.

    Expands ``~``, resolves symlinks, requires an existing regular file, and
    ensures the resolved path is contained in one of ``allowed_roots``.
    """
    if file_path is None or str(file_path).strip() == "":
        raise IngestPathError(400, "invalid_path", "file_path is required")

    try:
        raw = os.path.expanduser(str(file_path).strip())
    except (TypeError, ValueError) as exc:
        raise IngestPathError(400, "invalid_path", "Invalid file_path") from exc

    if not raw:
        raise IngestPathError(400, "invalid_path", "Invalid file_path")

    if not allowed_roots:
        raise IngestPathError(403, "path_not_allowed", "No ingest roots configured")

    # Join a trusted realpath root with a relative segment, then startswith-check
    # before any filesystem probe (CodeQL-recognized allowlist pattern).
    soft = os.path.normpath(raw)
    for root in allowed_roots:
        try:
            base = os.path.realpath(os.path.expanduser(str(root)))
        except OSError:
            continue
        base_prefix = base if base.endswith(os.sep) else base + os.sep
        soft_root = os.path.normpath(os.path.expanduser(str(root)))
        soft_prefix = soft_root if soft_root.endswith(os.sep) else soft_root + os.sep

        if os.path.isabs(soft):
            # Prefer matching the caller path against the canonical root; also
            # accept the pre-realpath root spelling so symlink roots still work.
            if soft == base or soft == soft_root:
                rel = ""
            elif soft.startswith(base_prefix):
                rel = soft[len(base_prefix) :]
            elif soft.startswith(soft_prefix):
                rel = soft[len(soft_prefix) :]
            else:
                continue
        else:
            rel = soft

        if rel == ".." or rel.startswith(".." + os.sep) or f"{os.sep}..{os.sep}" in f"{os.sep}{rel}{os.sep}":
            continue

        joined = os.path.normpath(os.path.join(base, rel)) if rel else base
        if joined != base and not joined.startswith(base_prefix):
            continue

        try:
            resolved = os.path.realpath(joined)
        except OSError:
            continue
        if resolved != base and not resolved.startswith(base_prefix):
            continue

        if resolved == base:
            verified_rel = ""
        else:
            verified_rel = resolved[len(base_prefix) :]
            if not verified_rel or verified_rel == ".." or verified_rel.startswith(".." + os.sep):
                continue

        return _open_allowlisted_file(base, verified_rel)

    raise IngestPathError(403, "path_not_allowed", "file_path is outside allowed ingest roots")
