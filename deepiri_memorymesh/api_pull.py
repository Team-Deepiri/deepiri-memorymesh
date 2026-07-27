"""Safe, explicit, token-gated API pulling (T38).

This module only understands **generic** response shapes:

- ``json``   — a single JSON object/array of message-shaped dicts (same
  flexible shape as :func:`deepiri_memorymesh.providers.base.parse_generic_file`).
- ``jsonl``  — newline-delimited JSON, one message object per line.
- ``bundle`` — the portable ``{"project", "messages": [...]}`` shape produced
  by :meth:`MemoryMesh.export_bundle`.

It deliberately does **not** implement any vendor-specific chat-history API
(OpenAI/ChatGPT, Claude, etc.). ``--provider`` is only a label attached to the
ingested rows; it never selects vendor parsing logic.

Network safety (SSRF hardening) is centered on :func:`validate_pull_url`,
which:

- Only allows ``https://`` by default; ``http://`` requires ``--allow-private``
  (intended for loopback test servers only).
- Rejects URLs with embedded userinfo (``user:pass@host``).
- Rejects ``file:``/``data:``/``ftp:`` and any other non-http(s) scheme.
- Requires the destination hostname to appear in an explicit allowlist
  (``--allow-host``, merged with ``Settings.api_pull_allowed_hosts``).
- Resolves the hostname once via ``getaddrinfo`` and validates **every**
  returned address against private/loopback/link-local/multicast/reserved/
  unspecified ranges, rejecting the whole request if any address is
  disallowed (unless ``--allow-private`` is set).
- Connects directly to a validated resolved IP address (never re-resolves at
  connect time), while still presenting the original hostname for TLS SNI and
  certificate hostname verification.
- Never follows redirects.

Secrets: the bearer token is read from an environment variable (named by
``--token-env``) or a file (``--token-file``); it is never accepted as a
plain CLI string, never written to YAML/SQLite/reports/URLs, and is redacted
from any exception text this module raises.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import socket
import ssl
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

from .config import Settings
from .models import MemoryRecord, now_iso
from .providers.base import normalize_content, safe_str
from .storage import MemoryStore, with_busy_retry

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB
DEFAULT_PROVIDER_TAG = "jsonl"

SUPPORTED_FORMATS = ("json", "jsonl", "bundle")

_CHUNK_SIZE = 65536
_DRAIN_CAP_BYTES = 65536
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_TEMP_PREFIX = "memorymesh-apipull-"

_COMPATIBLE_CONTENT_TYPE_HINTS = ("json", "ndjson", "jsonl", "text/plain", "octet-stream")
_INCOMPATIBLE_CONTENT_TYPE_PREFIXES = ("text/html", "image/", "video/", "audio/")


class ApiPullError(RuntimeError):
    """Base class for all api-pull failures. Messages never contain secrets."""


class ApiPullPolicyError(ApiPullError):
    """Rejected before (or without) any network I/O: bad config or SSRF policy."""


class ApiPullNetworkError(ApiPullError):
    """Network/protocol failure. Message text is sanitized of the token."""


class ApiPullSizeLimitError(ApiPullNetworkError):
    """Response exceeded the configured byte budget."""


@dataclass(slots=True)
class ApiPullReport:
    """Per-call report. Never carries the token or the raw response body."""

    url: str
    project: str
    format: str
    provider: str
    status: int | None = None
    conditional: str = "full"
    bytes_received: int = 0
    seen: int = 0
    inserted: int = 0
    duplicates: int = 0
    failures: int = 0
    failure_details: list[str] = field(default_factory=list)
    content_type: str | None = None
    dry_run: bool = False


@dataclass(slots=True)
class _ValidatedTarget:
    scheme: str
    host: str
    port: int
    path_and_query: str
    sanitized_url: str
    url_hash: str
    resolved_addresses: list[str]


@dataclass(slots=True)
class _PriorState:
    etag: str | None
    last_modified: str | None


def _classify_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a rejection reason code for a disallowed address, else None."""
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_multicast:
        return "multicast"
    if ip.is_reserved:
        return "reserved"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link_local"
    if ip.is_private:
        return "private"
    return None


def _normalized_identity(scheme: str, host: str, port: int, path: str) -> str:
    """Scheme+host+path identity with default ports collapsed; no query/fragment."""
    default_port = 443 if scheme == "https" else 80
    port_part = "" if port == default_port else f":{port}"
    return f"{scheme}://{host}{port_part}{path or '/'}"


def url_identity_hash(url: str) -> str:
    """sha256 of the normalized URL identity (scheme+host+path; never the query)."""
    parsed = urlsplit(url.strip())
    host = (parsed.hostname or "").strip().lower()
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    identity = _normalized_identity(parsed.scheme, host, port, parsed.path or "/")
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def sanitize_url(url: str) -> str:
    """Sanitized URL for reports/logs: scheme+host+path only, no query/fragment."""
    parsed = urlsplit(url.strip())
    host = (parsed.hostname or "").strip().lower()
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return _normalized_identity(parsed.scheme, host, port, parsed.path or "/")


def validate_pull_url(
    url: str,
    *,
    allow_hosts: Iterable[str] = (),
    allow_private: bool = False,
) -> _ValidatedTarget:
    """Validate *url* against the SSRF policy and resolve it to safe addresses.

    Raises :class:`ApiPullPolicyError` for any policy violation (bad scheme,
    userinfo, host not allowlisted, disallowed address range) and
    :class:`ApiPullNetworkError` for DNS resolution failures. Performs no
    network I/O beyond DNS resolution.
    """
    if not url or not url.strip():
        raise ApiPullPolicyError("URL is required.")
    parsed = urlsplit(url.strip())

    if parsed.scheme not in ("http", "https"):
        raise ApiPullPolicyError(
            f"Unsupported URL scheme {parsed.scheme!r}; only http/https are allowed "
            "(file/data/ftp and others are rejected)."
        )
    if parsed.scheme == "http" and not allow_private:
        raise ApiPullPolicyError(
            "http:// URLs are only allowed with --allow-private (intended for "
            "loopback test servers). Use https:// for real endpoints."
        )
    if parsed.username or parsed.password:
        raise ApiPullPolicyError("URLs with embedded userinfo (user:pass@host) are rejected.")

    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ApiPullPolicyError("URL is missing a hostname.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    path_and_query = path + (f"?{parsed.query}" if parsed.query else "")

    allow_set = {h.strip().lower() for h in allow_hosts if h and h.strip()}
    if host not in allow_set:
        raise ApiPullPolicyError(
            f"Host {host!r} is not in the allowlist. Pass --allow-host {host} "
            "(repeatable) or add it to the configured allowlist to permit this "
            "destination."
        )

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ApiPullNetworkError(f"Could not resolve host {host!r}: {type(exc).__name__}") from exc
    if not infos:
        raise ApiPullNetworkError(f"Host {host!r} did not resolve to any address.")

    resolved: list[str] = []
    seen_ips: set[str] = set()
    for info in infos:
        ip_str = info[4][0]
        if "%" in ip_str:
            ip_str = ip_str.split("%", 1)[0]
        if ip_str in seen_ips:
            continue
        seen_ips.add(ip_str)
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise ApiPullPolicyError(
                f"Host {host!r} resolved to an unrecognized address {ip_str!r}."
            ) from exc
        reason = _classify_ip(ip)
        if reason is not None:
            permitted_when_private = reason in {"private", "loopback", "link_local"}
            if not (allow_private and permitted_when_private):
                raise ApiPullPolicyError(
                    f"Host {host!r} resolves to a {reason} address ({ip_str}); "
                    "refusing without --allow-private."
                )
        resolved.append(ip_str)

    if not resolved:
        raise ApiPullNetworkError(f"Host {host!r} has no usable resolved addresses.")

    sanitized = _normalized_identity(parsed.scheme, host, port, path)
    return _ValidatedTarget(
        scheme=parsed.scheme,
        host=host,
        port=port,
        path_and_query=path_and_query,
        sanitized_url=sanitized,
        url_hash=hashlib.sha256(sanitized.encode("utf-8")).hexdigest(),
        resolved_addresses=resolved,
    )


def settings_allowed_hosts(settings: Settings | None) -> list[str]:
    """Host allowlist configured in Settings (empty list if unset)."""
    if settings is None:
        return []
    return list(getattr(settings, "api_pull_allowed_hosts", []) or [])


def resolve_token(
    *,
    token_env: str | None = None,
    token_file: Path | str | None = None,
) -> str | None:
    """Resolve the bearer token from an env var or file. Never from a CLI value.

    Returns ``None`` when neither is provided (unauthenticated pull).
    """
    if token_env and token_file:
        raise ApiPullPolicyError("Specify only one of --token-env or --token-file, not both.")
    if token_env:
        raw = os.environ.get(token_env)
        if raw is None or not raw.strip():
            raise ApiPullPolicyError(
                f"Environment variable {token_env!r} is not set or is empty."
            )
        return raw.strip()
    if token_file:
        path = Path(token_file)
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ApiPullPolicyError(
                f"Could not read token file: {type(exc).__name__}"
            ) from exc
        token = raw_text.strip()
        if not token:
            raise ApiPullPolicyError("Token file is empty.")
        return token
    return None


def _safe_error_text(exc: BaseException, token: str | None) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if token:
        text = text.replace(token, "[REDACTED]")
    return text


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to a pre-validated IP, with SNI/hostname checks intact."""

    def __init__(self, host: str, port: int, resolved_ip: str, timeout: float, context: ssl.SSLContext):
        super().__init__(host, port=port, timeout=timeout, context=context)
        self._resolved_ip = resolved_ip

    def connect(self) -> None:  # noqa: D102 - overriding stdlib connect()
        sock = socket.create_connection((self._resolved_ip, self.port), timeout=self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Plain-HTTP connection pinned to a pre-validated IP (loopback testing only)."""

    def __init__(self, host: str, port: int, resolved_ip: str, timeout: float):
        super().__init__(host, port=port, timeout=timeout)
        self._resolved_ip = resolved_ip

    def connect(self) -> None:  # noqa: D102 - overriding stdlib connect()
        self.sock = socket.create_connection((self._resolved_ip, self.port), timeout=self.timeout)


def _open_connection(
    target: _ValidatedTarget, *, timeout: float
) -> http.client.HTTPConnection:
    last_exc: Exception | None = None
    for ip_str in target.resolved_addresses:
        try:
            if target.scheme == "https":
                context = ssl.create_default_context()
                conn: http.client.HTTPConnection = _PinnedHTTPSConnection(
                    target.host, target.port, ip_str, timeout, context
                )
            else:
                conn = _PinnedHTTPConnection(target.host, target.port, ip_str, timeout)
            conn.connect()
            return conn
        except OSError as exc:
            last_exc = exc
            continue
    reason = type(last_exc).__name__ if last_exc else "no_resolved_addresses"
    raise ApiPullNetworkError(f"Could not connect to {target.host!r}: {reason}")


def _drain_response(resp: http.client.HTTPResponse, *, cap: int = _DRAIN_CAP_BYTES) -> None:
    """Best-effort bounded drain so the connection can be reused/closed cleanly."""
    try:
        remaining = cap
        while remaining > 0:
            chunk = resp.read(min(_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
    except Exception:
        pass


def _perform_request(
    target: _ValidatedTarget,
    *,
    timeout: float,
    token: str | None,
    if_none_match: str | None,
    if_modified_since: str | None,
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    conn = _open_connection(target, timeout=timeout)
    headers = {
        "Accept": "application/json",
        "User-Agent": "deepiri-memorymesh-api-pull/1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if if_none_match:
        headers["If-None-Match"] = if_none_match
    if if_modified_since:
        headers["If-Modified-Since"] = if_modified_since
    try:
        conn.request("GET", target.path_and_query, headers=headers)
        resp = conn.getresponse()
    except Exception as exc:
        conn.close()
        raise ApiPullNetworkError(_safe_error_text(exc, token)) from exc
    return conn, resp


def _check_content_type(content_type: str | None, fmt: str) -> None:
    if not content_type:
        return
    lowered = content_type.lower()
    if any(hint in lowered for hint in _COMPATIBLE_CONTENT_TYPE_HINTS):
        return
    if lowered.startswith(_INCOMPATIBLE_CONTENT_TYPE_PREFIXES):
        raise ApiPullError(f"Unexpected content-type for {fmt} response: {content_type!r}")
    # Unrecognized-but-not-obviously-wrong content types are accepted; many
    # generic APIs omit or mislabel content-type on otherwise valid JSON.


def _download_to_tempfile(
    resp: http.client.HTTPResponse,
    *,
    max_bytes: int,
    token: str | None,
) -> tuple[Path, int]:
    """Stream *resp* into a mode-0600 temp file, enforcing *max_bytes*.

    Caller is responsible for deleting the returned path (use ``finally``).
    """
    fd, name = tempfile.mkstemp(prefix=_TEMP_PREFIX, suffix=".tmp")
    tmp_path = Path(name)
    try:
        os.chmod(name, 0o600)
    except OSError:
        pass
    total = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            while True:
                try:
                    chunk = resp.read(_CHUNK_SIZE)
                except Exception as exc:
                    raise ApiPullNetworkError(_safe_error_text(exc, token)) from exc
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ApiPullSizeLimitError(
                        f"Response exceeded max-bytes limit ({max_bytes} bytes)."
                    )
                handle.write(chunk)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return tmp_path, total


def _parse_jsonl_entries(raw_text: str) -> tuple[list[dict], int, int]:
    """Return (dict messages, malformed line count, total non-blank lines)."""
    messages: list[dict] = []
    malformed = 0
    total = 0
    for line in raw_text.splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(item, dict):
            messages.append(item)
        else:
            malformed += 1
    return messages, malformed, total


def _parse_json_entries(raw_text: str) -> tuple[str | None, list[dict]]:
    payload = json.loads(raw_text)
    if isinstance(payload, list):
        return None, [m for m in payload if isinstance(m, dict)]
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object or an array")
    conv_id: str | None = None
    for key in ("conversation_id", "id", "chat_id", "session_id"):
        val = payload.get(key)
        if val:
            conv_id = safe_str(val)
            break
    messages = (
        payload.get("messages")
        or payload.get("conversation")
        or payload.get("items")
        or payload.get("turns")
        or []
    )
    if not isinstance(messages, list):
        raise ValueError("'messages' must be a list")
    return conv_id, [m for m in messages if isinstance(m, dict)]


def _parse_bundle_entries(raw_text: str) -> tuple[str | None, list[dict]]:
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Bundle root must be a JSON object")
    messages = payload.get("messages")
    if messages is None:
        messages = []
    if not isinstance(messages, list):
        raise ValueError("Bundle 'messages' must be a list")
    return None, [m for m in messages if isinstance(m, dict)]


def _extract_entries(fmt: str, raw_text: str) -> tuple[str | None, list[dict], list[str], int]:
    """Return (fallback conversation_id, dict entries, diagnostics, extra malformed count)."""
    if fmt == "jsonl":
        messages, malformed_lines, _total = _parse_jsonl_entries(raw_text)
        diagnostics = []
        if malformed_lines:
            diagnostics.append(f"{malformed_lines} malformed JSONL line(s) skipped")
        return None, messages, diagnostics, malformed_lines
    if fmt == "json":
        try:
            conv_id, messages = _parse_json_entries(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ApiPullError(f"Malformed JSON response body: {type(exc).__name__}") from exc
        return conv_id, messages, [], 0
    if fmt == "bundle":
        try:
            conv_id, messages = _parse_bundle_entries(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ApiPullError(f"Malformed bundle response body: {type(exc).__name__}") from exc
        return conv_id, messages, [], 0
    raise ApiPullPolicyError(
        f"Unsupported format {fmt!r}; expected one of: {', '.join(SUPPORTED_FORMATS)}."
    )


def _entry_to_record(
    msg: dict,
    *,
    provider: str,
    project: str,
    fallback_conversation_id: str,
    ordinal: int,
    url_hash: str,
) -> MemoryRecord | None:
    role = safe_str(msg.get("role") or msg.get("author") or msg.get("speaker"), "unknown")
    content = normalize_content(
        msg.get("content") or msg.get("text") or msg.get("message") or msg.get("parts")
    )
    if not content:
        return None

    conversation_id = safe_str(msg.get("conversation_id"), "") or fallback_conversation_id
    timestamp = safe_str(
        msg.get("timestamp") or msg.get("created_at") or msg.get("time") or msg.get("date")
    ) or now_iso()

    metadata = msg.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    metadata = dict(metadata)
    metadata["api_pull_ordinal"] = ordinal
    metadata_json = json.dumps(metadata, ensure_ascii=True)

    message_id = msg.get("id") or msg.get("message_id") or msg.get("uuid")
    if message_id not in (None, ""):
        source_key = f"apipull:{url_hash}:mid:{safe_str(message_id)}"
    else:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        source_key = f"apipull:{url_hash}:{ordinal}:{role}:{digest[:16]}"

    return MemoryRecord(
        provider=provider,
        project=project,
        conversation_id=conversation_id,
        role=role,
        content=content,
        timestamp=timestamp,
        metadata_json=metadata_json,
        source_key=source_key,
    )


def _parse_response_body(
    *,
    fmt: str,
    raw_text: str,
    provider: str,
    project: str,
    url_hash: str,
) -> tuple[list[MemoryRecord], int, list[str]]:
    fallback_conversation_id = f"api-pull:{url_hash[:24]}"
    conv_id, raw_messages, diagnostics, extra_malformed = _extract_entries(fmt, raw_text)

    records: list[MemoryRecord] = []
    malformed = list(diagnostics)
    for ordinal, msg in enumerate(raw_messages, start=1):
        if not isinstance(msg, dict):
            malformed.append(f"entry {ordinal}: not an object")
            continue
        try:
            rec = _entry_to_record(
                msg,
                provider=provider,
                project=project,
                fallback_conversation_id=conv_id or fallback_conversation_id,
                ordinal=ordinal,
                url_hash=url_hash,
            )
        except Exception as exc:
            malformed.append(f"entry {ordinal}: {type(exc).__name__}")
            continue
        if rec is None:
            malformed.append(f"entry {ordinal}: empty content")
            continue
        records.append(rec)

    seen = len(raw_messages) + extra_malformed
    return records, seen, malformed


def _load_state(
    store: MemoryStore, *, url_hash: str, project: str, fmt: str
) -> _PriorState | None:
    with store.connection() as conn:
        row = conn.execute(
            """
            SELECT etag, last_modified FROM api_pull_state
            WHERE url_hash = ? AND project = ? AND format = ?
            """,
            (url_hash, project, fmt),
        ).fetchone()
    if row is None:
        return None
    return _PriorState(etag=row["etag"], last_modified=row["last_modified"])


def _save_state(
    store: MemoryStore,
    *,
    url_hash: str,
    project: str,
    fmt: str,
    provider: str,
    etag: str | None,
    last_modified: str | None,
) -> None:
    def _do() -> None:
        with store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO api_pull_state
                    (url_hash, project, format, provider, etag, last_modified, last_success_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url_hash, project, format) DO UPDATE SET
                    provider = excluded.provider,
                    etag = excluded.etag,
                    last_modified = excluded.last_modified,
                    last_success_at = excluded.last_success_at
                """,
                (url_hash, project, fmt, provider, etag, last_modified, now_iso()),
            )

    with_busy_retry(_do)


def pull_api(
    *,
    store: MemoryStore,
    settings: Settings | None,
    project: str,
    url: str,
    fmt: str,
    provider: str = DEFAULT_PROVIDER_TAG,
    token_env: str | None = None,
    token_file: Path | str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allow_hosts: Iterable[str] = (),
    allow_private: bool = False,
    dry_run: bool = False,
) -> ApiPullReport:
    """Pull one page from a generic JSON/JSONL/bundle API and ingest new messages.

    Idempotent and append-only: repeated calls against an unchanged response
    insert nothing new (matched via stable ``source_key`` identity). Never
    stores the bearer token anywhere. Always cleans up its temp file.
    """
    fmt_norm = (fmt or "").strip().lower()
    if fmt_norm not in SUPPORTED_FORMATS:
        raise ApiPullPolicyError(
            f"Unsupported format {fmt_norm!r}; expected one of: {', '.join(SUPPORTED_FORMATS)}."
        )
    project_name = (project or "").strip()
    if not project_name:
        raise ApiPullPolicyError("project is required.")
    provider_tag = (provider or DEFAULT_PROVIDER_TAG).strip().lower() or DEFAULT_PROVIDER_TAG

    token = resolve_token(token_env=token_env, token_file=token_file)

    merged_allow_hosts = set(allow_hosts) | set(settings_allowed_hosts(settings))
    target = validate_pull_url(url, allow_hosts=merged_allow_hosts, allow_private=allow_private)

    report = ApiPullReport(
        url=target.sanitized_url,
        project=project_name,
        format=fmt_norm,
        provider=provider_tag,
        dry_run=dry_run,
    )

    prior = _load_state(store, url_hash=target.url_hash, project=project_name, fmt=fmt_norm)
    if_none_match = prior.etag if prior else None
    if_modified_since = prior.last_modified if prior else None

    conn: http.client.HTTPConnection | None = None
    tmp_path: Path | None = None
    try:
        conn, resp = _perform_request(
            target,
            timeout=timeout,
            token=token,
            if_none_match=if_none_match,
            if_modified_since=if_modified_since,
        )
        report.status = resp.status

        if resp.status in _REDIRECT_STATUSES:
            _drain_response(resp)
            report.conditional = "redirect_rejected"
            report.failures += 1
            report.failure_details.append(
                f"redirects are disabled; server returned HTTP {resp.status}"
            )
            return report

        if resp.status == 304:
            report.conditional = "not_modified"
            _drain_response(resp)
            if not dry_run:
                _save_state(
                    store,
                    url_hash=target.url_hash,
                    project=project_name,
                    fmt=fmt_norm,
                    provider=provider_tag,
                    etag=resp.getheader("ETag") or (prior.etag if prior else None),
                    last_modified=resp.getheader("Last-Modified")
                    or (prior.last_modified if prior else None),
                )
            return report

        if resp.status != 200:
            report.conditional = "error"
            report.failures += 1
            report.failure_details.append(f"unexpected HTTP status {resp.status}")
            _drain_response(resp)
            return report

        content_type = resp.getheader("Content-Type")
        report.content_type = content_type
        _check_content_type(content_type, fmt_norm)

        declared_length = resp.getheader("Content-Length")
        if declared_length is not None:
            try:
                declared = int(declared_length)
            except ValueError:
                declared = None
            if declared is not None and declared > max_bytes:
                _drain_response(resp)
                raise ApiPullSizeLimitError(
                    f"Declared Content-Length {declared} exceeds max-bytes limit "
                    f"({max_bytes})."
                )

        tmp_path, total = _download_to_tempfile(resp, max_bytes=max_bytes, token=token)
        report.bytes_received = total

        raw_text = tmp_path.read_text(encoding="utf-8", errors="replace")
        records, seen, malformed = _parse_response_body(
            fmt=fmt_norm,
            raw_text=raw_text,
            provider=provider_tag,
            project=project_name,
            url_hash=target.url_hash,
        )
        report.seen = seen
        report.failures = len(malformed)
        report.failure_details = malformed

        if not dry_run:
            inserted = store.insert_messages(records) if records else 0
            report.inserted = inserted
            report.duplicates = max(0, len(records) - inserted)
            _save_state(
                store,
                url_hash=target.url_hash,
                project=project_name,
                fmt=fmt_norm,
                provider=provider_tag,
                etag=resp.getheader("ETag"),
                last_modified=resp.getheader("Last-Modified"),
            )

        return report
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if conn is not None:
            conn.close()


__all__ = [
    "ApiPullError",
    "ApiPullPolicyError",
    "ApiPullNetworkError",
    "ApiPullSizeLimitError",
    "ApiPullReport",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_PROVIDER_TAG",
    "SUPPORTED_FORMATS",
    "pull_api",
    "resolve_token",
    "sanitize_url",
    "url_identity_hash",
    "validate_pull_url",
    "settings_allowed_hosts",
]
