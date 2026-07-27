"""Focused regression tests for T02/T03/T08/T01/T31."""

from __future__ import annotations

import http.client
import json
import os
import socket
import sqlite3
import tempfile
import threading
import unittest
import uuid
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from deepiri_memorymesh import Memory
from deepiri_memorymesh.cli import app
from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.http_security import (
    MAX_REQUEST_BODY_BYTES,
    assert_loopback_host,
    bind_address_family,
    normalize_ingest_roots,
    validate_ingest_file_path,
    IngestPathError,
)
from deepiri_memorymesh.service_api import (
    MemoryMeshHandler,
    ThreadingHTTPServerV6,
    make_http_server,
    run_service,
)
from deepiri_memorymesh.sync_service import MemoryMesh


def _iso_has_timezone(value: str) -> bool:
    parsed = datetime.fromisoformat(value)
    return parsed.tzinfo is not None


def _gai_entry(ip: str, family: int | None = None):
    if family is None:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return (family, socket.SOCK_STREAM, 6, "", (ip, 0))


class MemoryEmbedderAndTimestampTests(unittest.TestCase):
    """T03 + T08."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "memory.db"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_fallback_store_and_query(self) -> None:
        mem = Memory(db_path=self.db_path, embedder="fallback")
        mem.store("user likes rust")
        results = mem.query("what does the user like?")
        self.assertIn("user likes rust", results)

    def test_invalid_embedder_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Memory(db_path=self.db_path, embedder="invalid-value")
        self.assertIn("Unsupported embedder", str(ctx.exception))

    def test_auto_works_when_sentence_transformers_unavailable(self) -> None:
        import sys
        import types

        fake = types.ModuleType("sentence_transformers")

        class _Boom:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("sentence-transformers unavailable for test")

        fake.SentenceTransformer = _Boom  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake}):
            mem = Memory(db_path=self.db_path, embedder="auto")
            mem.store("async refactor in progress")
            self.assertTrue(mem.query("async work"))

    def test_created_at_is_timezone_aware_iso_not_uuid(self) -> None:
        mem = Memory(db_path=self.db_path, embedder="fallback")
        mem.store("unique timestamped memory")
        conn = sqlite3.connect(self.db_path)
        created_at = conn.execute(
            "SELECT timestamp FROM memory_messages WHERE content = ?",
            ("unique timestamped memory",),
        ).fetchone()[0]
        conn.close()

        self.assertTrue(_iso_has_timezone(created_at))
        with self.assertRaises(ValueError):
            uuid.UUID(created_at)

    def test_duplicate_content_still_deduped(self) -> None:
        mem = Memory(db_path=self.db_path, embedder="fallback")
        mem.store("same content")
        mem.store("same content")
        self.assertEqual(mem.all(), ["same content"])


class LoopbackHostTests(unittest.TestCase):
    """T01 — bind host validation."""

    def test_loopback_ipv4_literal_accepted(self) -> None:
        assert_loopback_host("127.0.0.1")

    def test_localhost_resolves_via_getaddrinfo(self) -> None:
        with mock.patch(
            "deepiri_memorymesh.http_security.socket.getaddrinfo",
            wraps=socket.getaddrinfo,
        ) as gai:
            assert_loopback_host("localhost")
            self.assertTrue(gai.called)
            self.assertEqual(gai.call_args.args[0], "localhost")

    def test_hostname_resolving_only_to_loopback_accepted(self) -> None:
        with mock.patch(
            "deepiri_memorymesh.http_security.socket.getaddrinfo",
            return_value=[_gai_entry("127.0.0.1")],
        ) as gai:
            assert_loopback_host("loopback.only.test")
            gai.assert_called()

    def test_hostname_resolving_to_non_loopback_rejected(self) -> None:
        with mock.patch(
            "deepiri_memorymesh.http_security.socket.getaddrinfo",
            return_value=[
                _gai_entry("127.0.0.1"),
                _gai_entry("8.8.8.8"),
            ],
        ):
            with self.assertRaises(ValueError) as ctx:
                assert_loopback_host("mixed.example.test")
            self.assertIn("8.8.8.8", str(ctx.exception))

    def test_all_interfaces_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_loopback_host("0.0.0.0")
        with self.assertRaises(ValueError):
            assert_loopback_host("::")

    def test_external_address_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_loopback_host("8.8.8.8")
        with self.assertRaises(ValueError):
            assert_loopback_host("1.2.3.4")

    def test_ipv6_loopback_literal_accepted_by_helper(self) -> None:
        assert_loopback_host("::1")
        self.assertEqual(bind_address_family("::1"), socket.AF_INET6)


class RunServiceGuardTests(unittest.TestCase):
    """run_service must fail closed before constructing a server."""

    def test_run_service_rejects_non_loopback_before_server_construction(self) -> None:
        with mock.patch("deepiri_memorymesh.service_api.make_http_server") as make_server:
            with mock.patch("deepiri_memorymesh.service_api.MemoryMesh") as mesh_cls:
                with mock.patch("deepiri_memorymesh.service_api.Settings.load") as load:
                    with self.assertRaises(ValueError):
                        run_service(host="0.0.0.0", port=8765)
                    load.assert_not_called()
                    mesh_cls.assert_not_called()
                    make_server.assert_not_called()


class ServeCliIngestRootTests(unittest.TestCase):
    """Typer serve --ingest-root wiring."""

    def test_serve_ingest_root_option_and_repeated_values(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            root_a = Path(tmp) / "a"
            root_b = Path(tmp) / "b"
            root_a.mkdir()
            root_b.mkdir()
            settings = Settings(
                db_path=Path(tmp) / "mesh.db",
                embedding_backend="fallback",
                providers=["jsonl"],
                provider_paths={},
                provider_globs={},
            )
            with mock.patch("deepiri_memorymesh.cli.run_service") as run_mock:
                with mock.patch("deepiri_memorymesh.cli.Settings.load", return_value=settings):
                    result = runner.invoke(
                        app,
                        [
                            "serve",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            "8765",
                            "--ingest-root",
                            str(root_a),
                            "--ingest-root",
                            str(root_b),
                        ],
                    )
            self.assertEqual(result.exit_code, 0, msg=result.output)
            run_mock.assert_called_once()
            kwargs = run_mock.call_args.kwargs
            roots = [Path(p).resolve() for p in kwargs["ingest_roots"]]
            self.assertIn(root_a.resolve(), roots)
            self.assertIn(root_b.resolve(), roots)


class IngestPathValidationTests(unittest.TestCase):
    """T01 — file_path allowlist."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name).resolve()
        self.allowed = self.root / "allowed"
        self.allowed.mkdir()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.allowed_file = self.allowed / "ok.json"
        self.allowed_file.write_text(
            json.dumps({"conversation_id": "c1", "messages": [{"role": "user", "content": "hi"}]}),
            encoding="utf-8",
        )
        self.outside_file = self.outside / "secret.json"
        self.outside_file.write_text(
            json.dumps({"conversation_id": "c2", "messages": [{"role": "user", "content": "nope"}]}),
            encoding="utf-8",
        )
        self.roots = normalize_ingest_roots(
            provider_paths={},
            extra_roots=[self.allowed],
            transfer_dir=self.root / "transfers",
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_file_under_allowed_root_accepted(self) -> None:
        path = validate_ingest_file_path(self.allowed_file, self.roots)
        self.assertEqual(path, self.allowed_file.resolve())

    def test_file_outside_roots_rejected(self) -> None:
        with self.assertRaises(IngestPathError) as ctx:
            validate_ingest_file_path(self.outside_file, self.roots)
        self.assertEqual(ctx.exception.status, 403)

    def test_traversal_rejected(self) -> None:
        traversal = self.allowed / ".." / "outside" / "secret.json"
        with self.assertRaises(IngestPathError) as ctx:
            validate_ingest_file_path(traversal, self.roots)
        self.assertEqual(ctx.exception.status, 403)

    def test_symlink_escaping_root_rejected(self) -> None:
        link = self.allowed / "escape.json"
        try:
            link.symlink_to(self.outside_file)
        except OSError as exc:  # pragma: no cover - platform may block symlinks
            self.skipTest(f"symlinks unavailable: {exc}")
        with self.assertRaises(IngestPathError) as ctx:
            validate_ingest_file_path(link, self.roots)
        self.assertEqual(ctx.exception.status, 403)

    def test_directory_rejected(self) -> None:
        with self.assertRaises(IngestPathError) as ctx:
            validate_ingest_file_path(self.allowed, self.roots)
        self.assertEqual(ctx.exception.status, 400)

    def test_missing_file_404(self) -> None:
        missing = self.allowed / "missing.json"
        with self.assertRaises(IngestPathError) as ctx:
            validate_ingest_file_path(missing, self.roots)
        self.assertEqual(ctx.exception.status, 404)


class _SilentHandler(MemoryMeshHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


class HttpServiceTests(unittest.TestCase):
    """T01 + T31 HTTP behavior."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name).resolve()
        self.home = self.root / "home"
        self.home.mkdir()
        self._old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.home)

        self.allowed = self.root / "allowed"
        self.allowed.mkdir()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.allowed_file = self.allowed / "chat.json"
        self.allowed_file.write_text(
            json.dumps(
                {
                    "conversation_id": "http-1",
                    "messages": [{"role": "user", "content": "from allowed file"}],
                }
            ),
            encoding="utf-8",
        )
        self.outside_file = self.outside / "chat.json"
        self.outside_file.write_text(
            json.dumps(
                {
                    "conversation_id": "http-2",
                    "messages": [{"role": "user", "content": "from outside file"}],
                }
            ),
            encoding="utf-8",
        )

        settings = Settings(
            db_path=self.root / "mesh.db",
            embedding_backend="fallback",
            providers=["jsonl"],
            provider_paths={"jsonl": str(self.allowed)},
            provider_globs={"jsonl": ["**/*.json"]},
        )
        self.mesh = MemoryMesh(settings)
        self.mesh.init()
        self.roots = normalize_ingest_roots(
            provider_paths=settings.provider_paths,
            extra_roots=[],
            transfer_dir=self.home / ".config" / "deepiri-memorymesh" / "transfers",
        )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SilentHandler)
        self.server.mesh = self.mesh  # type: ignore[attr-defined]
        self.server.ingest_roots = self.roots  # type: ignore[attr-defined]
        # T37 auth is out of scope for these tests; disable it.
        self.server.auth_mode = "off"  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._thread.join(timeout=5)
        if self._old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._old_home
        self._tmpdir.cleanup()

    def _post(self, path: str, payload: dict, headers: dict | None = None) -> tuple[int, dict]:
        body = json.dumps(payload).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        hdrs = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        if headers:
            hdrs.update(headers)
        conn.request("POST", path, body=body, headers=hdrs)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()
        return resp.status, json.loads(raw) if raw else {}

    def test_ingest_allowed_file(self) -> None:
        status, data = self._post(
            "/ingest",
            {
                "provider": "jsonl",
                "project": "p",
                "file_path": str(self.allowed_file),
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(data.get("ok"))
        self.assertGreaterEqual(int(data.get("inserted", 0)), 1)

    def test_ingest_outside_file_forbidden(self) -> None:
        status, data = self._post(
            "/ingest",
            {
                "provider": "jsonl",
                "project": "p",
                "file_path": str(self.outside_file),
            },
        )
        self.assertEqual(status, 403)
        self.assertFalse(data.get("ok"))

    def test_inline_conversation_still_works(self) -> None:
        status, data = self._post(
            "/ingest",
            {
                "provider": "jsonl",
                "project": "inline",
                "conversation": {
                    "conversation_id": "inline-1",
                    "messages": [{"role": "user", "content": "inline hello"}],
                },
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(data.get("ok"))
        self.assertGreaterEqual(int(data.get("inserted", 0)), 1)

    def test_cli_style_direct_ingest_unrestricted(self) -> None:
        # Direct library/CLI ingest does not use the HTTP allowlist.
        inserted = self.mesh.ingest_file("jsonl", "direct", self.outside_file)
        self.assertGreaterEqual(inserted, 1)

    def test_normal_json_below_limit(self) -> None:
        status, data = self._post("/query", {"project": "p", "q": "hello", "top_k": 3})
        self.assertEqual(status, 200)
        self.assertTrue(data.get("ok"))

    def test_unexpected_exception_returns_500_without_details(self) -> None:
        with mock.patch.object(
            self.mesh, "query_with_report", side_effect=RuntimeError("secret boom")
        ):
            status, data = self._post("/query", {"project": "p", "q": "hello", "top_k": 3})
        self.assertEqual(status, 500)
        self.assertEqual(data.get("error"), "internal_error")
        self.assertNotIn("secret boom", json.dumps(data))
        self.assertNotIn("Traceback", json.dumps(data))

    def test_body_exactly_at_limit_accepted(self) -> None:
        limit = 64
        prefix = b'{"q":"'
        suffix = b'"}'
        inner = b"z" * (limit - len(prefix) - len(suffix))
        raw = prefix + inner + suffix
        self.assertEqual(len(raw), limit)

        with mock.patch("deepiri_memorymesh.service_api.MAX_REQUEST_BODY_BYTES", limit):
            handler = MemoryMeshHandler.__new__(MemoryMeshHandler)
            handler.headers = {"Content-Length": str(limit)}  # type: ignore[attr-defined]
            handler.rfile = _BytesReader(raw)  # type: ignore[attr-defined]
            body = MemoryMeshHandler._json_body(handler)
            self.assertEqual(body.get("q"), inner.decode("ascii"))

    def test_body_one_byte_over_limit_413_without_reading(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        conn.putrequest("POST", "/ingest")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(MAX_REQUEST_BODY_BYTES + 1))
        conn.endheaders()
        # Deliberately send no body. If the handler tried to read Content-Length
        # bytes it would block until timeout.
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()
        self.assertEqual(resp.status, 413)
        data = json.loads(raw)
        self.assertEqual(data.get("error"), "payload_too_large")

    def test_missing_content_length_411(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", "/query")
        conn.putheader("Content-Type", "application/json")
        conn.endheaders()
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()
        self.assertEqual(resp.status, 411)
        self.assertEqual(json.loads(raw).get("error"), "length_required")

    def test_negative_content_length_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", "/query")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", "-1")
        conn.endheaders()
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()
        self.assertEqual(resp.status, 400)
        self.assertEqual(json.loads(raw).get("error"), "invalid_content_length")

    def test_nonnumeric_content_length_400(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.putrequest("POST", "/query")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", "abc")
        conn.endheaders()
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()
        self.assertEqual(resp.status, 400)
        self.assertEqual(json.loads(raw).get("error"), "invalid_content_length")

    def test_malformed_json_400(self) -> None:
        body = b"{not-json"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/query",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()
        self.assertEqual(resp.status, 400)
        data = json.loads(raw)
        self.assertEqual(data.get("error"), "invalid_json")
        self.assertNotIn("Traceback", raw)

    def test_json_array_400(self) -> None:
        body = b"[1,2,3]"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(
            "POST",
            "/query",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()
        self.assertEqual(resp.status, 400)
        self.assertEqual(json.loads(raw).get("error"), "invalid_json")


class IPv6LoopbackServiceTests(unittest.TestCase):
    """Real ::1 HTTP server behavior (not helper-only)."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.server = None
        self._thread = None
        settings = Settings(
            db_path=Path(self._tmpdir.name) / "mesh.db",
            embedding_backend="fallback",
            providers=["jsonl"],
            provider_paths={},
            provider_globs={},
        )
        self.mesh = MemoryMesh(settings)
        self.mesh.init()
        try:
            self.server = make_http_server("::1", 0, _SilentHandler)
        except OSError as exc:
            self._tmpdir.cleanup()
            self.skipTest(f"IPv6 ::1 bind unavailable: {exc}")
        self.assertIsInstance(self.server, ThreadingHTTPServerV6)
        self.server.mesh = self.mesh  # type: ignore[attr-defined]
        self.server.ingest_roots = []  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            if self._thread is not None:
                self._thread.join(timeout=5)
        self._tmpdir.cleanup()

    def test_health_over_ipv6_loopback(self) -> None:
        conn = http.client.HTTPConnection("::1", self.port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertTrue(json.loads(raw).get("ok"))


class _BytesReader:
    def __init__(self, data: bytes):
        self._data = data
        self.read_calls: list[int] = []

    def read(self, n: int = -1) -> bytes:
        self.read_calls.append(n)
        if n < 0:
            out = self._data
            self._data = b""
            return out
        out = self._data[:n]
        self._data = self._data[n:]
        return out


class BodyLimitUnitTests(unittest.TestCase):
    """T31 — ensure oversized Content-Length never triggers rfile.read."""

    def test_oversize_does_not_call_read(self) -> None:
        handler = MemoryMeshHandler.__new__(MemoryMeshHandler)
        handler.headers = {"Content-Length": str(MAX_REQUEST_BODY_BYTES + 1)}  # type: ignore[attr-defined]
        reader = _BytesReader(b"x")
        handler.rfile = reader  # type: ignore[attr-defined]
        with self.assertRaises(Exception) as ctx:
            MemoryMeshHandler._json_body(handler)
        self.assertEqual(ctx.exception.status, 413)  # type: ignore[attr-defined]
        self.assertEqual(reader.read_calls, [])


class PackagingSmokeTests(unittest.TestCase):
    """Dependency import surface used by the package and CLI."""

    def test_typer_and_yaml_importable_after_local_path_setup(self) -> None:
        import typer
        import yaml

        self.assertTrue(hasattr(typer, "Typer"))
        self.assertTrue(callable(yaml.safe_load))
        self.assertGreater(MAX_REQUEST_BODY_BYTES, 0)
        self.assertTrue(hasattr(Memory, "store"))


if __name__ == "__main__":
    unittest.main()
