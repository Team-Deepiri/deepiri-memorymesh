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


def validate_ingest_file_path(file_path: str | Path, allowed_roots: list[Path]) -> Path:
    """
    Validate a caller-supplied ingest path against allowed roots.

    Expands ``~``, resolves symlinks, requires an existing regular file, and
    ensures the resolved path is contained in one of ``allowed_roots``.
    """
    if file_path is None or str(file_path).strip() == "":
        raise IngestPathError(400, "invalid_path", "file_path is required")

    try:
        expanded = os.path.expanduser(str(file_path))
    except (TypeError, ValueError) as exc:
        raise IngestPathError(400, "invalid_path", "Invalid file_path") from exc

    if not expanded.strip():
        raise IngestPathError(400, "invalid_path", "Invalid file_path")

    try:
        resolved = os.path.realpath(expanded)
    except OSError as exc:
        raise IngestPathError(400, "invalid_path", "Invalid file_path") from exc

    if not allowed_roots:
        raise IngestPathError(403, "path_not_allowed", "No ingest roots configured")

    # Containment check first so path I/O only runs on allowlisted paths.
    sanitized: str | None = None
    for root in allowed_roots:
        try:
            root_real = os.path.realpath(str(root))
        except OSError:
            continue
        # Trailing sep so /allowed is not a prefix of /allowed-elsewhere.
        root_prefix = root_real if root_real.endswith(os.sep) else root_real + os.sep
        if resolved == root_real or resolved.startswith(root_prefix):
            sanitized = resolved
            break

    if sanitized is None:
        raise IngestPathError(403, "path_not_allowed", "file_path is outside allowed ingest roots")

    if not os.path.isfile(sanitized):
        if not os.path.exists(sanitized):
            raise IngestPathError(404, "path_not_found", "Ingest file not found")
        raise IngestPathError(400, "invalid_path", "file_path must be a regular file")

    return Path(sanitized)
