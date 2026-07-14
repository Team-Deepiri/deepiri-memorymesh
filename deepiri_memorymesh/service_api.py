from __future__ import annotations

import json
import socket
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .auth import verify_bearer
from .config import Settings
from .http_security import (
    MAX_REQUEST_BODY_BYTES,
    HttpApiError,
    assert_loopback_host,
    bind_address_family,
    normalize_ingest_roots,
    validate_ingest_file_path,
)
from .sync_service import MemoryMesh

AUTH_MODES = frozenset({"required", "off"})
DEFAULT_AUTH_MODE = "required"


class ThreadingHTTPServerV6(ThreadingHTTPServer):
    """IPv6-capable threaded HTTP server for binding ``::1``."""

    address_family = socket.AF_INET6


def make_http_server(
    host: str,
    port: int,
    handler: type[BaseHTTPRequestHandler],
) -> ThreadingHTTPServer:
    """Construct an IPv4 or IPv6 ``ThreadingHTTPServer`` for ``host``."""
    family = bind_address_family(host)
    server_cls: type[ThreadingHTTPServer]
    if family == socket.AF_INET6:
        server_cls = ThreadingHTTPServerV6
    else:
        server_cls = ThreadingHTTPServer
    return server_cls((host, port), handler)


class MemoryMeshHandler(BaseHTTPRequestHandler):
    server_version = "DeepiriMemoryMesh/0.1"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None or str(raw_len).strip() == "":
            raise HttpApiError(
                HTTPStatus.LENGTH_REQUIRED,
                "length_required",
                "Content-Length header is required",
            )

        text = str(raw_len).strip()
        if text.startswith("-") or not text.isdigit():
            raise HttpApiError(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Invalid Content-Length header",
            )

        length = int(text)
        if length > MAX_REQUEST_BODY_BYTES:
            raise HttpApiError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "payload_too_large",
                f"Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes",
            )

        data = self.rfile.read(length) if length > 0 else b""
        if not data:
            raise HttpApiError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "Request body must be a JSON object",
            )

        try:
            parsed = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpApiError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "Malformed JSON body",
            ) from exc

        if not isinstance(parsed, dict):
            raise HttpApiError(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "Request body must be a JSON object",
            )
        return parsed

    @property
    def mesh(self) -> MemoryMesh:
        return self.server.mesh  # type: ignore[attr-defined]

    @property
    def ingest_roots(self) -> list[Path]:
        return list(getattr(self.server, "ingest_roots", []))

    @property
    def auth_mode(self) -> str:
        return str(getattr(self.server, "auth_mode", DEFAULT_AUTH_MODE))

    def _require_scope(self, project: str, required_scope: str) -> None:
        """Enforce bearer auth for a project-scoped endpoint.

        Default-deny: any endpoint that calls this must present a valid
        bearer token for *project* with *required_scope*, unless the server
        was explicitly started with ``auth_mode="off"``. Never queries the
        URL for a token — only ``Authorization: Bearer`` is accepted.
        """
        if self.auth_mode == "off":
            return
        header = self.headers.get("Authorization")
        result = verify_bearer(
            self.mesh.settings.db_path,
            header,
            project=project,
            required_scope=required_scope,
        )
        if not result.ok:
            raise HttpApiError(result.status, result.error or "unauthorized", result.message or "Unauthorized")

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path == "/health":
                payload = getattr(self.server, "health_payload", None)
                if isinstance(payload, dict):
                    self._send(HTTPStatus.OK, payload)
                    return
                # Fallback identity for ``memorymesh serve`` (not TUI-owned).
                mesh = self.mesh
                try:
                    status = mesh.store.database_status()
                    schema_version = status.schema_version
                except Exception:
                    schema_version = 0
                from .supervised_service import database_identity

                self._send(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "service": "memorymesh",
                        "api_version": "0.1",
                        "schema_version": schema_version,
                        "instance_id": getattr(self.server, "instance_id", "serve"),
                        "db_identity": database_identity(mesh.settings.db_path),
                    },
                )
                return
            if self.path.startswith("/stats"):
                qs = self.path.split("?", 1)[1] if "?" in self.path else ""
                project = "default"
                for part in qs.split("&"):
                    if part.startswith("project="):
                        project = part.split("=", 1)[1] or "default"
                self._require_scope(project, "read")
                self._send(HTTPStatus.OK, {"ok": True, "stats": self.mesh.stats(project)})
                return
            self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
        except HttpApiError as exc:
            self._send(exc.status, exc.payload())
        except Exception:
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": "internal_error", "message": "Request failed"},
            )

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._json_body()
            if self.path == "/ingest":
                provider = str(body.get("provider") or "unknown")
                project = str(body.get("project") or "default")
                self._require_scope(project, "write")
                file_path = body.get("file_path")
                if file_path is not None and str(file_path).strip() != "":
                    safe_path = validate_ingest_file_path(str(file_path), self.ingest_roots)
                    inserted = self.mesh.ingest_file(provider, project, safe_path)
                    self._send(HTTPStatus.OK, {"ok": True, "inserted": inserted})
                    return

                conversation = body.get("conversation") or {}
                if not isinstance(conversation, dict):
                    raise HttpApiError(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_json",
                        "conversation must be a JSON object",
                    )
                conv_id = str(conversation.get("conversation_id") or "api-conversation")
                messages = conversation.get("messages") or []
                payload = {
                    "conversation_id": conv_id,
                    "messages": messages,
                }
                tmp_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
                        tmp.write(json.dumps(payload, ensure_ascii=True))
                        tmp_path = Path(tmp.name)
                    inserted = self.mesh.ingest_file(provider, project, tmp_path)
                finally:
                    if tmp_path is not None:
                        tmp_path.unlink(missing_ok=True)
                self._send(HTTPStatus.OK, {"ok": True, "inserted": inserted})
                return

            if self.path == "/query":
                project = str(body.get("project") or "default")
                self._require_scope(project, "read")
                text = str(body.get("q") or "")
                top_k = int(body.get("top_k") or 8)
                strategy = body.get("strategy") or body.get("mode")
                candidate_limit = body.get("candidate_limit")
                result = self.mesh.query_with_report(
                    project,
                    text,
                    top_k=top_k,
                    strategy=str(strategy) if strategy is not None else None,
                    candidate_limit=int(candidate_limit) if candidate_limit is not None else None,
                )
                report = result.report
                payload = {
                    "ok": True,
                    "results": result.rows,
                    "scored_versioned": report.scored_versioned,
                    "legacy_compatible_embeddings": report.legacy_compatible,
                    "skipped_incompatible": report.skipped_incompatible,
                    "skipped_malformed": report.skipped_malformed,
                    # Aggregates kept for older clients.
                    "skipped_embeddings": report.skipped,
                    "scored_embeddings": report.scored,
                    "strategy_requested": report.strategy_requested,
                    "strategy_used": report.strategy_used,
                    "total_eligible_embeddings": report.total_eligible_embeddings,
                    "candidate_message_count": report.candidate_message_count,
                    "embeddings_scored": report.embeddings_scored,
                    "candidate_limit": report.candidate_limit,
                }
                if report.exact_fallback_reason:
                    payload["exact_fallback_reason"] = report.exact_fallback_reason
                if result.diagnostic:
                    payload["diagnostic"] = result.diagnostic
                self._send(HTTPStatus.OK, payload)
                return

            if self.path == "/state/put":
                project = str(body.get("project") or "default")
                self._require_scope(project, "write")
                self.mesh.put_state(
                    project=project,
                    agent=str(body.get("agent") or "unknown"),
                    key=str(body.get("key") or ""),
                    value=str(body.get("value") or ""),
                )
                self._send(HTTPStatus.OK, {"ok": True})
                return

            if self.path == "/state/get":
                project = str(body.get("project") or "default")
                self._require_scope(project, "read")
                value = self.mesh.get_state(
                    project=project,
                    agent=str(body.get("agent") or "unknown"),
                    key=str(body.get("key") or ""),
                )
                self._send(HTTPStatus.OK, {"ok": True, "value": value})
                return

            self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
        except HttpApiError as exc:
            self._send(exc.status, exc.payload())
        except Exception:
            # Do not leak internal exception details to clients.
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": "internal_error", "message": "Request failed"},
            )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Redact bearer material that may appear in request lines (e.g. clients
        # mistakenly putting tokens in query strings). Never log Authorization.
        safe_args = []
        for arg in args:
            text = str(arg)
            if "mmht." in text:
                import re

                text = re.sub(r"mmht\.[A-Za-z0-9._\-]+", "mmht.[REDACTED]", text)
            if "token=" in text.lower():
                import re

                text = re.sub(r"(?i)token=[^&\s]+", "token=[REDACTED]", text)
            safe_args.append(text)
        super().log_message(format, *safe_args)


def run_service(
    host: str = "127.0.0.1",
    port: int = 8765,
    ingest_roots: list[Path] | None = None,
    settings: Settings | None = None,
    auth_mode: str = DEFAULT_AUTH_MODE,
) -> None:
    assert_loopback_host(host)
    if auth_mode not in AUTH_MODES:
        raise ValueError(f"auth_mode must be one of {sorted(AUTH_MODES)}, got {auth_mode!r}")
    cfg = settings if settings is not None else Settings.load()
    mesh = MemoryMesh(cfg)
    mesh.init()
    roots = ingest_roots
    if roots is None:
        roots = normalize_ingest_roots(
            provider_paths=cfg.provider_paths,
            extra_roots=None,
        )
    server = make_http_server(host, port, MemoryMeshHandler)
    server.mesh = mesh  # type: ignore[attr-defined]
    server.ingest_roots = list(roots)  # type: ignore[attr-defined]
    server.instance_id = "serve"  # type: ignore[attr-defined]
    server.auth_mode = auth_mode  # type: ignore[attr-defined]
    from .supervised_service import build_health_payload

    server.health_payload = build_health_payload(mesh, "serve")  # type: ignore[attr-defined]
    if auth_mode == "off":
        print(
            "WARNING: memorymesh service starting with authentication DISABLED "
            "(auth_mode=off). Any local process/user can read and write every "
            "project via this HTTP API. Use only for local development."
        )
    print(f"memorymesh service listening on http://{host}:{port} (auth_mode={auth_mode})")
    server.serve_forever()
