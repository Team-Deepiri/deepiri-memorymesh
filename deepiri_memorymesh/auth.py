"""Project-scoped HTTP bearer token authentication/authorization (T37).

Uses the ``project_http_tokens`` table (migration v5, see :mod:`migrations`).
Deliberately has **no** dependency on the optional ``cryptography`` package —
only stdlib ``hashlib``/``hmac``/``secrets`` are used.

Token format
------------
``mmht.<token_id>.<secret>`` where:

- ``token_id``: 16 hex chars (64 bits), non-secret, used as the primary key
  and the public identifier shown in ``auth token list``.
- ``secret``: 32 bytes of randomness, urlsafe-base64 encoded (256 bits).

Only ``token_id`` and a SHA-256 hash of the **full** presented token string
are ever persisted — the plaintext token is shown to the caller exactly once
(at creation/rotation time) and never stored, logged, or echoed again.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .migrations import DEFAULT_BUSY_TIMEOUT_MS, ensure_migrated

TOKEN_PREFIX = "mmht"
TOKEN_ID_BYTES = 8  # -> 16 hex chars
SECRET_BYTES = 32  # -> 256 bits
VALID_SCOPES = frozenset({"read", "write"})

# token_id: exactly 16 lowercase hex chars. secret: urlsafe-base64 alphabet,
# generous bounds so future secret lengths still parse.
_TOKEN_RE = re.compile(r"^mmht\.([0-9a-f]{16})\.([A-Za-z0-9_-]{16,128})$")

_BEARER_PREFIX = "bearer "


class AuthError(RuntimeError):
    """Base class for authentication/authorization failures in this module."""


class TokenNotFoundError(AuthError):
    """Raised by revoke/rotate when *token_id* has no matching row."""


class InvalidScopeError(AuthError, ValueError):
    """Raised for an empty or unrecognized scope list."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _new_token_id() -> str:
    return secrets.token_hex(TOKEN_ID_BYTES)


def _new_secret() -> str:
    return secrets.token_urlsafe(SECRET_BYTES)


def _hash_token(token_plaintext: str) -> str:
    return hashlib.sha256(token_plaintext.encode("utf-8")).hexdigest()


def format_token(token_id: str, secret: str) -> str:
    return f"{TOKEN_PREFIX}.{token_id}.{secret}"


def parse_token(token_plaintext: str) -> tuple[str, str] | None:
    """Return ``(token_id, secret)`` or ``None`` if the format doesn't match."""
    if not isinstance(token_plaintext, str):
        return None
    match = _TOKEN_RE.match(token_plaintext.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _normalize_scopes(scopes: Sequence[str]) -> tuple[str, ...]:
    cleaned = {str(s).strip().lower() for s in scopes if str(s).strip()}
    if not cleaned:
        raise InvalidScopeError("At least one scope is required (read and/or write).")
    unknown = cleaned - VALID_SCOPES
    if unknown:
        raise InvalidScopeError(
            f"Unknown scope(s) {sorted(unknown)!r}; valid scopes are {sorted(VALID_SCOPES)}."
        )
    return tuple(sorted(cleaned))


def _table_names(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {str(r[0]) for r in cur.fetchall()}


def _open_rw(db_path: Path, *, busy_timeout_ms: int) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_path), timeout=max(busy_timeout_ms / 1000.0, 0.001))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    return conn


def _open_ro(db_path: Path, *, busy_timeout_ms: int) -> sqlite3.Connection:
    from .storage import sqlite_readonly_uri

    conn = sqlite3.connect(
        sqlite_readonly_uri(Path(db_path)), uri=True, timeout=busy_timeout_ms / 1000.0
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


@dataclass(slots=True, frozen=True)
class TokenRecord:
    """Non-secret token metadata. Never contains the token or its hash."""

    token_id: str
    project: str
    label: str | None
    scopes: tuple[str, ...]
    created_at: str
    expires_at: str | None
    revoked_at: str | None
    last_used_at: str | None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


def _record_from_row(row: sqlite3.Row) -> TokenRecord:
    try:
        scopes = tuple(json.loads(row["scopes"]))
    except (json.JSONDecodeError, TypeError, ValueError):
        scopes = ()
    return TokenRecord(
        token_id=str(row["token_id"]),
        project=str(row["project"]),
        label=row["label"],
        scopes=scopes,
        created_at=str(row["created_at"]),
        expires_at=row["expires_at"],
        revoked_at=row["revoked_at"],
        last_used_at=row["last_used_at"],
    )


@dataclass(slots=True, frozen=True)
class AuthStatus:
    """Read-only auth diagnostics for one database (never mutates it)."""

    db_path: Path
    schema_ready: bool
    total_tokens: int
    active_tokens: int
    revoked_tokens: int
    projects: tuple[str, ...]


def auth_status(
    db_path: Path,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> AuthStatus:
    """Read-only token counts/diagnostics. Never writes anything."""
    path = Path(db_path)
    empty = AuthStatus(
        db_path=path, schema_ready=False, total_tokens=0, active_tokens=0,
        revoked_tokens=0, projects=(),
    )
    if not path.exists() or path.stat().st_size == 0:
        return empty
    conn = _open_ro(path, busy_timeout_ms=busy_timeout_ms)
    try:
        if "project_http_tokens" not in _table_names(conn):
            return empty
        total = int(conn.execute("SELECT COUNT(*) FROM project_http_tokens").fetchone()[0])
        active = int(
            conn.execute(
                "SELECT COUNT(*) FROM project_http_tokens WHERE revoked_at IS NULL"
            ).fetchone()[0]
        )
        projects = tuple(
            str(r[0])
            for r in conn.execute(
                "SELECT DISTINCT project FROM project_http_tokens ORDER BY project ASC"
            ).fetchall()
        )
        return AuthStatus(
            db_path=path,
            schema_ready=True,
            total_tokens=total,
            active_tokens=active,
            revoked_tokens=total - active,
            projects=projects,
        )
    finally:
        conn.close()


def create_token(
    db_path: Path,
    project: str,
    scopes: Sequence[str],
    *,
    label: str | None = None,
    expires_at: str | None = None,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> tuple[str, TokenRecord]:
    """Create a new bearer token for *project*. Returns ``(plaintext, record)``.

    The plaintext token is returned exactly once here; it cannot be recovered
    afterward (only its SHA-256 hash is stored). Raises
    :class:`InvalidScopeError` for empty/unknown scopes and :class:`ValueError`
    for a malformed *expires_at*.
    """
    proj = str(project).strip()
    if not proj:
        raise ValueError("project must be a non-empty string.")
    normalized_scopes = _normalize_scopes(scopes)
    if expires_at is not None:
        _parse_iso(expires_at)  # validate format; raises ValueError if bad

    path = Path(db_path)
    ensure_migrated(path, busy_timeout_ms=busy_timeout_ms)

    conn = _open_rw(path, busy_timeout_ms=busy_timeout_ms)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            created_at = _utc_now()
            scopes_json = json.dumps(list(normalized_scopes))
            for _ in range(5):
                token_id = _new_token_id()
                secret = _new_secret()
                token_plaintext = format_token(token_id, secret)
                token_hash = _hash_token(token_plaintext)
                try:
                    conn.execute(
                        """
                        INSERT INTO project_http_tokens
                            (token_id, project, token_hash, label, scopes,
                             created_at, expires_at, revoked_at, last_used_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                        """,
                        (token_id, proj, token_hash, label, scopes_json, created_at, expires_at),
                    )
                    break
                except sqlite3.IntegrityError:
                    continue  # token_id collision (astronomically unlikely); retry
            else:
                raise AuthError("Could not allocate a unique token_id after several attempts.")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    record = TokenRecord(
        token_id=token_id,
        project=proj,
        label=label,
        scopes=normalized_scopes,
        created_at=created_at,
        expires_at=expires_at,
        revoked_at=None,
        last_used_at=None,
    )
    return token_plaintext, record


def list_tokens(
    db_path: Path,
    project: str | None = None,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> list[TokenRecord]:
    """List tokens (optionally scoped to *project*). Never includes secrets."""
    path = Path(db_path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    conn = _open_ro(path, busy_timeout_ms=busy_timeout_ms)
    try:
        if "project_http_tokens" not in _table_names(conn):
            return []
        if project is None:
            rows = conn.execute(
                "SELECT * FROM project_http_tokens ORDER BY created_at DESC, token_id ASC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM project_http_tokens WHERE project = ? "
                "ORDER BY created_at DESC, token_id ASC",
                (str(project),),
            ).fetchall()
        return [_record_from_row(r) for r in rows]
    finally:
        conn.close()


def revoke_token(
    db_path: Path,
    token_id: str,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> TokenRecord:
    """Revoke a token by id. Idempotent: revoking an already-revoked token is a no-op.

    Raises :class:`TokenNotFoundError` if *token_id* doesn't exist.
    """
    path = Path(db_path)
    conn = _open_rw(path, busy_timeout_ms=busy_timeout_ms)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT * FROM project_http_tokens WHERE token_id = ?", (token_id,)
            ).fetchone()
            if row is None:
                raise TokenNotFoundError(f"No token found with id {token_id!r}.")
            if row["revoked_at"] is None:
                conn.execute(
                    "UPDATE project_http_tokens SET revoked_at = ? WHERE token_id = ?",
                    (_utc_now(), token_id),
                )
                row = conn.execute(
                    "SELECT * FROM project_http_tokens WHERE token_id = ?", (token_id,)
                ).fetchone()
            conn.commit()
            return _record_from_row(row)
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()


def rotate_token(
    db_path: Path,
    token_id: str,
    *,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> tuple[str, TokenRecord]:
    """Revoke *token_id* and create a fresh token with the same project/scopes/label.

    Returns ``(new_plaintext, new_record)``. Raises :class:`TokenNotFoundError`
    if *token_id* doesn't exist. Atomic: either both the revoke and the new
    insert happen, or neither does.
    """
    path = Path(db_path)
    conn = _open_rw(path, busy_timeout_ms=busy_timeout_ms)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            old_row = conn.execute(
                "SELECT * FROM project_http_tokens WHERE token_id = ?", (token_id,)
            ).fetchone()
            if old_row is None:
                raise TokenNotFoundError(f"No token found with id {token_id!r}.")
            old = _record_from_row(old_row)
            if old.revoked_at is None:
                conn.execute(
                    "UPDATE project_http_tokens SET revoked_at = ? WHERE token_id = ?",
                    (_utc_now(), token_id),
                )

            created_at = _utc_now()
            scopes_json = json.dumps(list(old.scopes))
            for _ in range(5):
                new_id = _new_token_id()
                secret = _new_secret()
                new_plaintext = format_token(new_id, secret)
                new_hash = _hash_token(new_plaintext)
                try:
                    conn.execute(
                        """
                        INSERT INTO project_http_tokens
                            (token_id, project, token_hash, label, scopes,
                             created_at, expires_at, revoked_at, last_used_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                        """,
                        (new_id, old.project, new_hash, old.label, scopes_json, created_at, old.expires_at),
                    )
                    break
                except sqlite3.IntegrityError:
                    continue
            else:
                raise AuthError("Could not allocate a unique token_id after several attempts.")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()

    new_record = TokenRecord(
        token_id=new_id,
        project=old.project,
        label=old.label,
        scopes=old.scopes,
        created_at=created_at,
        expires_at=old.expires_at,
        revoked_at=None,
        last_used_at=None,
    )
    return new_plaintext, new_record


def write_token_file(path: Path, token_plaintext: str, *, overwrite: bool = False) -> Path:
    """Write a plaintext token to *path* with mode 0o600.

    Mirrors :func:`crypto.generate_key_file`'s safety properties: refuses to
    overwrite an existing file unless *overwrite* is ``True``, and never
    returns/logs the token itself.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    try:
        fd = os.open(file_path, flags, 0o600)
    except FileExistsError as exc:
        raise AuthError(
            f"Token file already exists: {file_path}. Pass overwrite=True to replace it."
        ) from exc
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token_plaintext + "\n")
    finally:
        pass
    os.chmod(file_path, 0o600)
    return file_path


@dataclass(slots=True, frozen=True)
class VerificationResult:
    """Outcome of :func:`verify_bearer`. Never contains the presented token."""

    ok: bool
    status: int
    error: str | None = None
    message: str | None = None
    token_id: str | None = None
    project: str | None = None
    scopes: tuple[str, ...] = ()


_GENERIC_UNAUTHORIZED = "Missing or invalid bearer token."
_GENERIC_FORBIDDEN = "Token does not authorize this request."


def _deny_unauthorized(error: str) -> VerificationResult:
    return VerificationResult(ok=False, status=401, error=error, message=_GENERIC_UNAUTHORIZED)


def _deny_forbidden(error: str) -> VerificationResult:
    return VerificationResult(ok=False, status=403, error=error, message=_GENERIC_FORBIDDEN)


def _touch_last_used(db_path: Path, token_id: str, *, busy_timeout_ms: int) -> None:
    """Best-effort ``last_used_at`` update; failures are swallowed."""
    try:
        from .storage import with_busy_retry

        def _do() -> None:
            conn = _open_rw(db_path, busy_timeout_ms=busy_timeout_ms)
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "UPDATE project_http_tokens SET last_used_at = ? WHERE token_id = ?",
                    (_utc_now(), token_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        with_busy_retry(_do, attempts=2)
    except Exception:
        pass


def verify_bearer(
    db_path: Path,
    authorization_header: str | None,
    *,
    project: str,
    required_scope: str,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
) -> VerificationResult:
    """Verify an ``Authorization: Bearer <token>`` header for *project*/*required_scope*.

    Returns a :class:`VerificationResult`; never raises for ordinary
    authentication failures (missing/malformed/expired/revoked/wrong
    project/insufficient scope all come back as a non-``ok`` result with an
    appropriate HTTP status). Any unexpected internal error also fails
    *closed* (401) rather than allowing the request through. The presented
    token is never included in the result, in exceptions, or logged.
    """
    if required_scope not in VALID_SCOPES:
        raise ValueError(f"required_scope must be one of {sorted(VALID_SCOPES)}, got {required_scope!r}")

    try:
        if not authorization_header or not isinstance(authorization_header, str):
            return _deny_unauthorized("missing_token")

        header = authorization_header.strip()
        if not header.lower().startswith(_BEARER_PREFIX):
            return _deny_unauthorized("missing_token")

        presented = header[len(_BEARER_PREFIX):].strip()
        if not presented:
            return _deny_unauthorized("missing_token")

        parsed = parse_token(presented)
        if parsed is None:
            return _deny_unauthorized("invalid_token")
        token_id, _secret = parsed

        path = Path(db_path)
        if not path.exists() or path.stat().st_size == 0:
            return _deny_unauthorized("invalid_token")

        conn = _open_ro(path, busy_timeout_ms=busy_timeout_ms)
        try:
            if "project_http_tokens" not in _table_names(conn):
                return _deny_unauthorized("invalid_token")
            row = conn.execute(
                "SELECT token_hash, project, scopes, expires_at, revoked_at "
                "FROM project_http_tokens WHERE token_id = ?",
                (token_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return _deny_unauthorized("invalid_token")

        expected_hash = _hash_token(presented)
        if not hmac.compare_digest(expected_hash, str(row["token_hash"])):
            return _deny_unauthorized("invalid_token")

        if row["revoked_at"] is not None:
            return _deny_unauthorized("invalid_token")

        expires_at = row["expires_at"]
        if expires_at:
            try:
                if _parse_iso(str(expires_at)) <= datetime.now(timezone.utc):
                    return _deny_unauthorized("invalid_token")
            except ValueError:
                return _deny_unauthorized("invalid_token")

        try:
            scopes = tuple(json.loads(row["scopes"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            scopes = ()

        token_project = str(row["project"])
        if token_project != str(project):
            return _deny_forbidden("wrong_project")

        if required_scope not in scopes:
            return _deny_forbidden("insufficient_scope")

        _touch_last_used(path, token_id, busy_timeout_ms=busy_timeout_ms)

        return VerificationResult(
            ok=True,
            status=200,
            token_id=token_id,
            project=token_project,
            scopes=scopes,
        )
    except Exception:
        # Fail closed: any unexpected error must never be treated as "auth off".
        return _deny_unauthorized("internal_error")
