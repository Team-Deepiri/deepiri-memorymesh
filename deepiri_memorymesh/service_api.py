from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import Settings
from .sync_service import MemoryMesh


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
        raw_len = self.headers.get("Content-Length", "0")
        length = int(raw_len) if raw_len.isdigit() else 0
        if length <= 0:
            return {}
        data = self.rfile.read(length).decode("utf-8")
        return json.loads(data) if data else {}

    @property
    def mesh(self) -> MemoryMesh:
        return self.server.mesh  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(HTTPStatus.OK, {"ok": True, "service": "memorymesh"})
            return
        if self.path.startswith("/stats"):
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            project = "default"
            for part in qs.split("&"):
                if part.startswith("project="):
                    project = part.split("=", 1)[1] or "default"
            self._send(HTTPStatus.OK, {"ok": True, "stats": self.mesh.stats(project)})
            return
        self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._json_body()
            if self.path == "/ingest":
                provider = str(body.get("provider") or "unknown")
                project = str(body.get("project") or "default")
                file_path = body.get("file_path")
                if file_path:
                    from pathlib import Path

                    inserted = self.mesh.ingest_file(provider, project, Path(str(file_path)))
                    self._send(HTTPStatus.OK, {"ok": True, "inserted": inserted})
                    return
                conversation = body.get("conversation") or {}
                conv_id = str(conversation.get("conversation_id") or "api-conversation")
                messages = conversation.get("messages") or []
                payload = {
                    "conversation_id": conv_id,
                    "messages": messages,
                }
                from pathlib import Path
                import tempfile

                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
                    tmp.write(json.dumps(payload, ensure_ascii=True))
                    tmp_path = Path(tmp.name)
                inserted = self.mesh.ingest_file(provider, project, tmp_path)
                tmp_path.unlink(missing_ok=True)
                self._send(HTTPStatus.OK, {"ok": True, "inserted": inserted})
                return

            if self.path == "/query":
                project = str(body.get("project") or "default")
                text = str(body.get("q") or "")
                top_k = int(body.get("top_k") or 8)
                rows = self.mesh.query(project, text, top_k=top_k)
                self._send(HTTPStatus.OK, {"ok": True, "results": rows})
                return

            if self.path == "/state/put":
                self.mesh.put_state(
                    project=str(body.get("project") or "default"),
                    agent=str(body.get("agent") or "unknown"),
                    key=str(body.get("key") or ""),
                    value=str(body.get("value") or ""),
                )
                self._send(HTTPStatus.OK, {"ok": True})
                return

            if self.path == "/state/get":
                value = self.mesh.get_state(
                    project=str(body.get("project") or "default"),
                    agent=str(body.get("agent") or "unknown"),
                    key=str(body.get("key") or ""),
                )
                self._send(HTTPStatus.OK, {"ok": True, "value": value})
                return

            self._send(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
        except Exception as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})


def run_service(host: str = "127.0.0.1", port: int = 8765) -> None:
    settings = Settings.load()
    mesh = MemoryMesh(settings)
    mesh.init()
    server = ThreadingHTTPServer((host, port), MemoryMeshHandler)
    server.mesh = mesh  # type: ignore[attr-defined]
    print(f"memorymesh service listening on http://{host}:{port}")
    server.serve_forever()
