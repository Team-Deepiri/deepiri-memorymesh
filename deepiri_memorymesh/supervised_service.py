"""Supervised in-process HTTP service lifecycle (T34).

Plain TUI does not start a service. Optional ``--with-service`` owns a threaded
server that is always stopped in ``finally``. Never uses ``start_new_session``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .config import Settings
from .http_security import assert_loopback_host, normalize_ingest_roots
from .service_api import DEFAULT_AUTH_MODE, MemoryMeshHandler, make_http_server
from .sync_service import MemoryMesh

SERVICE_NAME = "memorymesh"
SERVICE_API_VERSION = "0.1"


def database_identity(db_path: Path | str) -> str:
    """Stable non-secret database identity (truncated SHA-256 of normalized path)."""
    try:
        normalized = str(Path(db_path).expanduser())
    except Exception:
        normalized = str(db_path)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass(slots=True)
class ServiceIdentity:
    service: str
    api_version: str
    schema_version: int
    instance_id: str
    db_identity: str


@dataclass(slots=True)
class HealthProbeResult:
    ok: bool
    compatible: bool
    identity: ServiceIdentity | None = None
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def build_health_payload(mesh: MemoryMesh, instance_id: str) -> dict[str, Any]:
    status = mesh.store.database_status()
    return {
        "ok": True,
        "service": SERVICE_NAME,
        "api_version": SERVICE_API_VERSION,
        "schema_version": status.schema_version,
        "instance_id": instance_id,
        "db_identity": database_identity(mesh.settings.db_path),
    }


def probe_health(host: str, port: int, *, timeout: float = 0.5) -> HealthProbeResult:
    """Probe loopback health without owning or stopping the remote service."""
    url = f"http://{host}:{port}/health"
    try:
        with urlopen(url, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        return HealthProbeResult(ok=False, compatible=False, error=f"url_error:{exc}")
    except Exception as exc:
        return HealthProbeResult(ok=False, compatible=False, error=f"{type(exc).__name__}")

    if not isinstance(raw, dict) or not raw.get("ok"):
        return HealthProbeResult(
            ok=False,
            compatible=False,
            error="invalid_health",
            raw=raw if isinstance(raw, dict) else {},
        )
    service = str(raw.get("service") or "")
    if service != SERVICE_NAME:
        return HealthProbeResult(
            ok=True,
            compatible=False,
            error="unrelated_service",
            raw=raw,
        )
    identity = ServiceIdentity(
        service=service,
        api_version=str(raw.get("api_version") or ""),
        schema_version=int(raw.get("schema_version") or 0),
        instance_id=str(raw.get("instance_id") or ""),
        db_identity=str(raw.get("db_identity") or ""),
    )
    compatible = identity.api_version == SERVICE_API_VERSION
    return HealthProbeResult(ok=True, compatible=compatible, identity=identity, raw=raw)


@dataclass
class SupervisedService:
    """Owns an in-process Memory Mesh HTTP server thread."""

    host: str
    port: int
    settings: Settings
    auth_mode: str = DEFAULT_AUTH_MODE
    instance_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    server: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None
    owned: bool = False
    start_result: str = ""
    reused_existing: bool = False

    def start(self) -> str:
        """Start or reuse a compatible service. Returns a short status token."""
        assert_loopback_host(self.host)
        existing = probe_health(self.host, self.port)
        if existing.ok and existing.compatible:
            self.owned = False
            self.reused_existing = True
            self.start_result = "reused_existing"
            return self.start_result
        if existing.ok and not existing.compatible:
            self.owned = False
            self.start_result = "port_conflict_unrelated"
            return self.start_result

        mesh = MemoryMesh(self.settings)
        mesh.init()
        roots = normalize_ingest_roots(
            provider_paths=self.settings.provider_paths,
            extra_roots=None,
        )

        class _Handler(MemoryMeshHandler):
            pass

        try:
            server = make_http_server(self.host, self.port, _Handler)
        except OSError:
            # Race: another service bound the port. Re-probe.
            time.sleep(0.05)
            raced = probe_health(self.host, self.port)
            if raced.ok and raced.compatible:
                self.owned = False
                self.reused_existing = True
                self.start_result = "reused_existing_after_race"
                return self.start_result
            self.owned = False
            self.start_result = "port_conflict"
            return self.start_result

        server.mesh = mesh  # type: ignore[attr-defined]
        server.ingest_roots = list(roots)  # type: ignore[attr-defined]
        server.instance_id = self.instance_id  # type: ignore[attr-defined]
        server.auth_mode = self.auth_mode  # type: ignore[attr-defined]
        server.health_payload = build_health_payload(mesh, self.instance_id)  # type: ignore[attr-defined]

        thread = threading.Thread(target=server.serve_forever, name="memorymesh-supervised", daemon=True)
        self.server = server
        self.thread = thread
        self.owned = True
        thread.start()

        # Wait briefly for health.
        for _ in range(40):
            probe = probe_health(self.host, self.port)
            if probe.ok and probe.compatible:
                self.start_result = "started"
                return self.start_result
            time.sleep(0.05)
        self.shutdown()
        self.start_result = "start_failed"
        return self.start_result

    def shutdown(self) -> None:
        """Stop only a service owned by this handle."""
        if not self.owned:
            return
        server = self.server
        thread = self.thread
        self.owned = False
        if server is not None:
            try:
                server.shutdown()
            except Exception:
                pass
            try:
                server.server_close()
            except Exception:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self.server = None
        self.thread = None

    def __enter__(self) -> "SupervisedService":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


def detect_existing_service(host: str = "127.0.0.1", port: int = 8765) -> HealthProbeResult:
    """Non-mutating health detection for TUI reuse messaging."""
    return probe_health(host, port)
