"""Batch 6: T38 safe, explicit, token-gated API pulling.

Uses a local ``127.0.0.1`` HTTP server (never external network) to exercise
the network-safety policy, secret handling, conditional requests, and
idempotent/append-only ingest behavior of :mod:`deepiri_memorymesh.api_pull`.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from unittest import mock

from tests.helpers import temp_home

from deepiri_memorymesh.api_pull import (
    ApiPullNetworkError,
    ApiPullPolicyError,
    ApiPullSizeLimitError,
    DEFAULT_MAX_BYTES,
    pull_api,
    resolve_token,
    sanitize_url,
    url_identity_hash,
    validate_pull_url,
)
from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.storage import MemoryStore

TOKEN_ENV_VAR = "MEMORYMESH_TEST_API_PULL_TOKEN"


class LocalApiServer:
    """A tiny local HTTP server with per-test-configurable routes."""

    def __init__(self) -> None:
        self.routes: dict[str, Callable[[BaseHTTPRequestHandler], None]] = {}
        self.received_headers: list[dict[str, str]] = []
        self.request_count = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:  # noqa: A003
                return

            def do_GET(self) -> None:  # noqa: N802
                outer.request_count += 1
                outer.received_headers.append(dict(self.headers.items()))
                path = self.path.split("?", 1)[0]
                fn = outer.routes.get(path)
                if fn is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                fn(self)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _send_json(handler: BaseHTTPRequestHandler, payload: object, *, status: int = 200, extra_headers: dict | None = None) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    for k, v in (extra_headers or {}).items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(body)


def _send_text(handler: BaseHTTPRequestHandler, body: bytes, *, status: int = 200, content_type: str = "application/json", extra_headers: dict | None = None) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    for k, v in (extra_headers or {}).items():
        handler.send_header(k, v)
    handler.end_headers()
    handler.wfile.write(body)


def _temp_files_with_prefix() -> list[str]:
    tmpdir = tempfile.gettempdir()
    return [n for n in os.listdir(tmpdir) if n.startswith("memorymesh-apipull-")]


class ApiPullTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.home = self.root / "home"
        self._home_ctx = temp_home(self.home)
        self._home_ctx.__enter__()
        self.db_path = self.root / "mesh.db"
        self.store = MemoryStore(self.db_path)
        self.store.init()
        self.settings = Settings(db_path=self.db_path)
        self.server = LocalApiServer()
        os.environ.pop(TOKEN_ENV_VAR, None)

    def tearDown(self) -> None:
        self.server.stop()
        os.environ.pop(TOKEN_ENV_VAR, None)
        self._home_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def _pull(self, **kwargs):
        kwargs.setdefault("store", self.store)
        kwargs.setdefault("settings", self.settings)
        kwargs.setdefault("project", "proj")
        kwargs.setdefault("allow_private", True)
        kwargs.setdefault("allow_hosts", ["127.0.0.1"])
        return pull_api(**kwargs)

    def _message_rows(self) -> list[dict]:
        with self.store.connection() as conn:
            rows = conn.execute(
                "SELECT provider, project, conversation_id, role, content, source_key "
                "FROM memory_messages ORDER BY id ASC"
            ).fetchall()
        return [dict(r) for r in rows]


class UrlPolicyTests(unittest.TestCase):
    def test_rejects_unsupported_scheme(self) -> None:
        for bad in ("file:///etc/passwd", "ftp://example.com/x", "data:text/plain,hi"):
            with self.assertRaises(ApiPullPolicyError):
                validate_pull_url(bad, allow_hosts=["example.com"], allow_private=True)

    def test_rejects_http_without_allow_private(self) -> None:
        with self.assertRaises(ApiPullPolicyError):
            validate_pull_url(
                "http://127.0.0.1:9/x", allow_hosts=["127.0.0.1"], allow_private=False
            )

    def test_rejects_userinfo(self) -> None:
        with self.assertRaises(ApiPullPolicyError):
            validate_pull_url(
                "https://user:pass@example.com/x",
                allow_hosts=["example.com"],
                allow_private=False,
            )

    def test_rejects_loopback_by_default_even_over_https(self) -> None:
        with self.assertRaises(ApiPullPolicyError):
            validate_pull_url(
                "https://127.0.0.1:9443/x", allow_hosts=["127.0.0.1"], allow_private=False
            )

    def test_allow_private_permits_loopback(self) -> None:
        target = validate_pull_url(
            "https://127.0.0.1:9443/x", allow_hosts=["127.0.0.1"], allow_private=True
        )
        self.assertIn("127.0.0.1", target.resolved_addresses)

    def test_requires_allowlisted_host(self) -> None:
        with self.assertRaises(ApiPullPolicyError):
            validate_pull_url(
                "https://127.0.0.1:9443/x", allow_hosts=["someone-else.example"], allow_private=True
            )

    def test_no_dns_resolution_for_disallowed_host(self) -> None:
        """Proves we never even attempt to resolve/connect to a non-allowlisted host."""
        with mock.patch("socket.getaddrinfo") as mocked:
            with self.assertRaises(ApiPullPolicyError):
                validate_pull_url(
                    "https://example.com/x", allow_hosts=["other.example"], allow_private=False
                )
            mocked.assert_not_called()

    def test_sanitized_url_drops_query_and_secrets(self) -> None:
        sanitized = sanitize_url("https://example.com/api/messages?api_key=super-secret&x=1")
        self.assertNotIn("super-secret", sanitized)
        self.assertNotIn("?", sanitized)
        self.assertEqual(sanitized, "https://example.com/api/messages")

    def test_url_identity_hash_stable_and_query_independent(self) -> None:
        h1 = url_identity_hash("https://example.com/api/messages?page=1")
        h2 = url_identity_hash("https://example.com/api/messages?page=2")
        h3 = url_identity_hash("https://example.com/api/messages")
        self.assertEqual(h1, h2)
        self.assertEqual(h1, h3)


class TokenResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop(TOKEN_ENV_VAR, None)

    def tearDown(self) -> None:
        os.environ.pop(TOKEN_ENV_VAR, None)

    def test_no_token_sources_returns_none(self) -> None:
        self.assertIsNone(resolve_token(token_env=None, token_file=None))

    def test_rejects_both_sources(self) -> None:
        with self.assertRaises(ApiPullPolicyError):
            resolve_token(token_env=TOKEN_ENV_VAR, token_file="/tmp/does-not-matter")

    def test_env_missing_raises(self) -> None:
        with self.assertRaises(ApiPullPolicyError):
            resolve_token(token_env=TOKEN_ENV_VAR, token_file=None)

    def test_env_token_resolved(self) -> None:
        os.environ[TOKEN_ENV_VAR] = "secret-abc"
        self.assertEqual(resolve_token(token_env=TOKEN_ENV_VAR, token_file=None), "secret-abc")

    def test_file_token_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "token.txt"
            path.write_text("secret-from-file\n", encoding="utf-8")
            self.assertEqual(
                resolve_token(token_env=None, token_file=path), "secret-from-file"
            )

    def test_empty_file_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "token.txt"
            path.write_text("   \n", encoding="utf-8")
            with self.assertRaises(ApiPullPolicyError):
                resolve_token(token_env=None, token_file=path)


class RedirectAndSchemeTests(ApiPullTestBase):
    def test_redirect_is_not_followed(self) -> None:
        def redirect(h: BaseHTTPRequestHandler) -> None:
            h.send_response(302)
            h.send_header("Location", "/elsewhere")
            h.send_header("Content-Length", "0")
            h.end_headers()

        self.server.routes["/redirect"] = redirect
        report = self._pull(
            url=f"{self.server.base_url}/redirect", fmt="json", provider="jsonl"
        )
        self.assertEqual(report.conditional, "redirect_rejected")
        self.assertEqual(report.inserted, 0)
        self.assertEqual(self._message_rows(), [])

    def test_non_200_status_recorded_without_writes(self) -> None:
        def broken(h: BaseHTTPRequestHandler) -> None:
            h.send_response(500)
            h.send_header("Content-Length", "0")
            h.end_headers()

        self.server.routes["/broken"] = broken
        report = self._pull(url=f"{self.server.base_url}/broken", fmt="json")
        self.assertEqual(report.status, 500)
        self.assertEqual(report.inserted, 0)
        self.assertEqual(self._message_rows(), [])


class SizeLimitTests(ApiPullTestBase):
    def test_declared_content_length_over_limit_rejected(self) -> None:
        big_payload = {"messages": [{"role": "user", "content": "x" * 1000}] * 50}
        body = json.dumps(big_payload).encode("utf-8")
        self.assertGreater(len(body), 100)

        def big(h: BaseHTTPRequestHandler) -> None:
            _send_text(h, body)

        self.server.routes["/big"] = big
        before = set(_temp_files_with_prefix())
        with self.assertRaises(ApiPullSizeLimitError):
            self._pull(url=f"{self.server.base_url}/big", fmt="json", max_bytes=100)
        after = set(_temp_files_with_prefix())
        self.assertEqual(before, after, "temp file must be cleaned up on size-limit rejection")
        self.assertEqual(self._message_rows(), [])

    def test_streamed_over_limit_without_content_length_rejected(self) -> None:
        big_payload = {"messages": [{"role": "user", "content": "y" * 1000}] * 50}
        body = json.dumps(big_payload).encode("utf-8")

        def chunked(h: BaseHTTPRequestHandler) -> None:
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Transfer-Encoding", "chunked")
            h.end_headers()
            chunk = body
            h.wfile.write(f"{len(chunk):x}\r\n".encode("ascii"))
            h.wfile.write(chunk)
            h.wfile.write(b"\r\n0\r\n\r\n")

        self.server.routes["/chunked"] = chunked
        before = set(_temp_files_with_prefix())
        with self.assertRaises(ApiPullSizeLimitError):
            self._pull(url=f"{self.server.base_url}/chunked", fmt="json", max_bytes=100)
        after = set(_temp_files_with_prefix())
        self.assertEqual(before, after)


class TokenHandlingLiveTests(ApiPullTestBase):
    def test_authorization_header_received_from_env_token(self) -> None:
        os.environ[TOKEN_ENV_VAR] = "s3cr3t-env-token"

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, {"messages": [{"role": "user", "content": "hi"}]})

        self.server.routes["/pull"] = handler
        report = self._pull(
            url=f"{self.server.base_url}/pull", fmt="json", token_env=TOKEN_ENV_VAR
        )
        self.assertEqual(report.inserted, 1)
        self.assertEqual(
            self.server.received_headers[-1].get("Authorization"), "Bearer s3cr3t-env-token"
        )

    def test_authorization_header_received_from_file_token(self) -> None:
        token_path = self.root / "token.txt"
        token_path.write_text("s3cr3t-file-token\n", encoding="utf-8")

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, {"messages": [{"role": "user", "content": "hi"}]})

        self.server.routes["/pull"] = handler
        report = self._pull(
            url=f"{self.server.base_url}/pull", fmt="json", token_file=token_path
        )
        self.assertEqual(report.inserted, 1)
        self.assertEqual(
            self.server.received_headers[-1].get("Authorization"), "Bearer s3cr3t-file-token"
        )

    def test_token_never_stored_in_db_or_report(self) -> None:
        os.environ[TOKEN_ENV_VAR] = "s3cr3t-must-not-persist"

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, {"messages": [{"role": "user", "content": "hi"}]})

        self.server.routes["/pull"] = handler
        report = self._pull(
            url=f"{self.server.base_url}/pull", fmt="json", token_env=TOKEN_ENV_VAR
        )
        report_text = json.dumps(
            {
                "url": report.url,
                "status": report.status,
                "conditional": report.conditional,
                "failure_details": report.failure_details,
            }
        )
        self.assertNotIn("s3cr3t-must-not-persist", report_text)

        raw_db_bytes = self.db_path.read_bytes()
        self.assertNotIn(b"s3cr3t-must-not-persist", raw_db_bytes)

    def test_token_redacted_from_network_error_text(self) -> None:
        os.environ[TOKEN_ENV_VAR] = "s3cr3t-in-error-path"
        # Port 1 on loopback should reliably refuse the connection.
        with self.assertRaises(ApiPullNetworkError) as ctx:
            self._pull(
                url="https://127.0.0.1:1/nope", fmt="json", token_env=TOKEN_ENV_VAR
            )
        self.assertNotIn("s3cr3t-in-error-path", str(ctx.exception))


class ConditionalRequestTests(ApiPullTestBase):
    def test_etag_conditional_304_is_noop(self) -> None:
        state = {"calls": 0}

        def handler(h: BaseHTTPRequestHandler) -> None:
            state["calls"] += 1
            if h.headers.get("If-None-Match") == '"v1"':
                h.send_response(304)
                h.send_header("Content-Length", "0")
                h.end_headers()
                return
            _send_json(
                h,
                {"messages": [{"role": "user", "content": "hello"}]},
                extra_headers={"ETag": '"v1"'},
            )

        self.server.routes["/etag"] = handler
        first = self._pull(url=f"{self.server.base_url}/etag", fmt="json")
        self.assertEqual(first.conditional, "full")
        self.assertEqual(first.inserted, 1)

        second = self._pull(url=f"{self.server.base_url}/etag", fmt="json")
        self.assertEqual(second.conditional, "not_modified")
        self.assertEqual(second.status, 304)
        self.assertEqual(second.inserted, 0)
        self.assertEqual(len(self._message_rows()), 1)
        self.assertEqual(state["calls"], 2)


class IdempotencyAndAppendOnlyTests(ApiPullTestBase):
    def test_repeated_identical_pull_is_idempotent(self) -> None:
        payload = {"messages": [{"id": "m1", "role": "user", "content": "hello"}]}

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, payload)

        self.server.routes["/msgs"] = handler
        first = self._pull(url=f"{self.server.base_url}/msgs", fmt="json")
        self.assertEqual(first.inserted, 1)
        self.assertEqual(first.duplicates, 0)

        second = self._pull(url=f"{self.server.base_url}/msgs", fmt="json")
        self.assertEqual(second.inserted, 0)
        self.assertEqual(second.duplicates, 1)
        self.assertEqual(len(self._message_rows()), 1)

    def test_append_only_new_message_added_on_growth(self) -> None:
        payload = {"messages": [{"id": "m1", "role": "user", "content": "hello"}]}

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, payload)

        self.server.routes["/grow"] = handler
        first = self._pull(url=f"{self.server.base_url}/grow", fmt="json")
        self.assertEqual(first.inserted, 1)

        payload["messages"].append({"id": "m2", "role": "assistant", "content": "world"})
        second = self._pull(url=f"{self.server.base_url}/grow", fmt="json")
        self.assertEqual(second.inserted, 1)
        self.assertEqual(second.duplicates, 1)
        rows = self._message_rows()
        self.assertEqual(len(rows), 2)

    def test_message_id_used_for_stable_source_key(self) -> None:
        payload = {"messages": [{"id": "abc-123", "role": "user", "content": "hello"}]}

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, payload)

        self.server.routes["/ids"] = handler
        self._pull(url=f"{self.server.base_url}/ids", fmt="json")
        rows = self._message_rows()
        self.assertEqual(len(rows), 1)
        self.assertIn(":mid:abc-123", rows[0]["source_key"])

    def test_ordinal_fallback_when_no_id_present(self) -> None:
        payload = {"messages": [{"role": "user", "content": "no id here"}]}

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, payload)

        self.server.routes["/noids"] = handler
        self._pull(url=f"{self.server.base_url}/noids", fmt="json")
        rows = self._message_rows()
        self.assertEqual(len(rows), 1)
        self.assertNotIn(":mid:", rows[0]["source_key"])
        self.assertIn(":1:user:", rows[0]["source_key"])


class FormatParsingTests(ApiPullTestBase):
    def test_json_object_with_messages(self) -> None:
        payload = {
            "conversation_id": "conv-1",
            "messages": [
                {"role": "user", "content": "hi there"},
                {"role": "assistant", "content": "hello!"},
            ],
        }

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, payload)

        self.server.routes["/json"] = handler
        report = self._pull(url=f"{self.server.base_url}/json", fmt="json")
        self.assertEqual(report.seen, 2)
        self.assertEqual(report.inserted, 2)
        rows = self._message_rows()
        self.assertEqual({r["conversation_id"] for r in rows}, {"conv-1"})

    def test_jsonl_lines(self) -> None:
        lines = [
            json.dumps({"role": "user", "content": "line one"}),
            json.dumps({"role": "assistant", "content": "line two"}),
        ]
        body = ("\n".join(lines) + "\n").encode("utf-8")

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_text(h, body, content_type="application/x-ndjson")

        self.server.routes["/jsonl"] = handler
        report = self._pull(url=f"{self.server.base_url}/jsonl", fmt="jsonl", provider="jsonl")
        self.assertEqual(report.inserted, 2)

    def test_bundle_shape(self) -> None:
        payload = {
            "project": "ignored-in-bundle-body",
            "messages": [
                {
                    "provider": "bundle-src",
                    "conversation_id": "conv-b",
                    "role": "user",
                    "content": "bundled message",
                }
            ],
        }

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, payload)

        self.server.routes["/bundle"] = handler
        report = self._pull(url=f"{self.server.base_url}/bundle", fmt="bundle")
        self.assertEqual(report.inserted, 1)
        rows = self._message_rows()
        self.assertEqual(rows[0]["conversation_id"], "conv-b")
        # Provider tag on the row comes from --provider, not the payload body.
        self.assertEqual(rows[0]["provider"], "jsonl")


class MalformedIsolationTests(ApiPullTestBase):
    def test_malformed_jsonl_lines_isolated(self) -> None:
        body = (
            json.dumps({"role": "user", "content": "good one"})
            + "\n"
            + "{not valid json"
            + "\n"
            + json.dumps({"role": "user", "content": "good two"})
            + "\n"
        ).encode("utf-8")

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_text(h, body)

        self.server.routes["/malformed"] = handler
        report = self._pull(url=f"{self.server.base_url}/malformed", fmt="jsonl")
        self.assertEqual(report.inserted, 2)
        self.assertGreaterEqual(report.failures, 1)
        self.assertTrue(any("malformed" in d for d in report.failure_details))

    def test_entries_missing_content_isolated_not_fatal(self) -> None:
        payload = {
            "messages": [
                {"role": "user", "content": "kept"},
                {"role": "user", "content": ""},
                {"role": "assistant"},
            ]
        }

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, payload)

        self.server.routes["/partial"] = handler
        report = self._pull(url=f"{self.server.base_url}/partial", fmt="json")
        self.assertEqual(report.inserted, 1)
        self.assertEqual(report.failures, 2)


class DryRunAndCleanupTests(ApiPullTestBase):
    def test_dry_run_does_not_write(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, payload)

        self.server.routes["/dry"] = handler
        report = self._pull(url=f"{self.server.base_url}/dry", fmt="json", dry_run=True)
        self.assertTrue(report.dry_run)
        self.assertEqual(report.seen, 1)
        self.assertEqual(report.inserted, 0)
        self.assertEqual(self._message_rows(), [])
        with self.store.connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM api_pull_state").fetchone()
        self.assertEqual(row["c"], 0)

    def test_temp_file_cleaned_up_on_success(self) -> None:
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, payload)

        self.server.routes["/clean"] = handler
        before = set(_temp_files_with_prefix())
        self._pull(url=f"{self.server.base_url}/clean", fmt="json")
        after = set(_temp_files_with_prefix())
        self.assertEqual(before, after)

    def test_temp_file_cleaned_up_on_parse_error(self) -> None:
        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_text(h, b"not json at all", content_type="application/json")

        self.server.routes["/badjson"] = handler
        before = set(_temp_files_with_prefix())
        with self.assertRaises(Exception):
            self._pull(url=f"{self.server.base_url}/badjson", fmt="json")
        after = set(_temp_files_with_prefix())
        self.assertEqual(before, after)


class NoExternalNetworkTests(ApiPullTestBase):
    def test_public_hostname_without_allowlist_never_resolves(self) -> None:
        with mock.patch("socket.getaddrinfo") as mocked:
            with self.assertRaises(ApiPullPolicyError):
                self._pull(
                    url="https://definitely-not-allowlisted.example/x",
                    fmt="json",
                    allow_hosts=[],
                    allow_private=False,
                )
            mocked.assert_not_called()

    def test_settings_allowlist_can_authorize_host(self) -> None:
        self.settings.api_pull_allowed_hosts = ["127.0.0.1"]

        def handler(h: BaseHTTPRequestHandler) -> None:
            _send_json(h, {"messages": [{"role": "user", "content": "hi"}]})

        self.server.routes["/via-settings"] = handler
        report = self._pull(
            url=f"{self.server.base_url}/via-settings",
            fmt="json",
            allow_hosts=[],
        )
        self.assertEqual(report.inserted, 1)


class DnsConnectionPinningTests(unittest.TestCase):
    def test_pinned_connect_uses_validated_ip_and_hostname(self) -> None:
        from deepiri_memorymesh import api_pull

        # Use clearly public unicast IPs (avoid TEST-NET docs ranges that some
        # Python versions classify as non-global / reserved / private).
        infos = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443)),
        ]
        connected: list[tuple[str, int]] = []
        wrap_hosts: list[str] = []

        class FakeSock:
            def close(self) -> None:
                return None

        class FakeSSLSock(FakeSock):
            def __init__(self, sock: FakeSock, *, server_hostname: str | None = None) -> None:
                wrap_hosts.append(server_hostname or "")

        class FakeContext:
            verify_mode = ssl.CERT_REQUIRED
            post_handshake_auth = None

            def set_alpn_protocols(self, protocols):  # noqa: ANN001
                return None

            def wrap_socket(self, sock: FakeSock, server_hostname: str | None = None) -> FakeSSLSock:
                return FakeSSLSock(sock, server_hostname=server_hostname)

        def fake_create_connection(address, timeout=None):  # noqa: ANN001
            connected.append(address)
            return FakeSock()

        with mock.patch("socket.getaddrinfo", return_value=infos):
            target = api_pull.validate_pull_url(
                "https://pin.example/path",
                allow_hosts=["pin.example"],
                allow_private=False,
            )
        self.assertEqual(target.resolved_addresses, ["8.8.8.8", "1.1.1.1"])

        with (
            mock.patch("socket.create_connection", side_effect=fake_create_connection),
            mock.patch("ssl.create_default_context", return_value=FakeContext()),
            mock.patch("socket.getaddrinfo") as second_resolve,
        ):
            conn = api_pull._open_connection(target, timeout=2.0)
            second_resolve.assert_not_called()
            self.assertIsInstance(conn, api_pull._PinnedHTTPSConnection)
            self.assertEqual(connected[0], ("8.8.8.8", 443))
            self.assertEqual(wrap_hosts[0], "pin.example")
            self.assertEqual(conn.host, "pin.example")

    def test_mixed_public_private_fail_closed(self) -> None:
        from deepiri_memorymesh import api_pull

        infos = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443)),
        ]
        with mock.patch("socket.getaddrinfo", return_value=infos):
            with self.assertRaises(ApiPullPolicyError) as ctx:
                api_pull.validate_pull_url(
                    "https://mixed.example/",
                    allow_hosts=["mixed.example"],
                    allow_private=False,
                )
        self.assertIn("private", str(ctx.exception).lower())

    def test_second_resolver_cannot_redirect_request(self) -> None:
        from deepiri_memorymesh import api_pull

        infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.4.4", 443))]
        calls: list[tuple[str, int]] = []

        class FakeSock:
            def close(self) -> None:
                return None

        class FakeContext:
            verify_mode = ssl.CERT_REQUIRED
            post_handshake_auth = None

            def set_alpn_protocols(self, protocols):  # noqa: ANN001
                return None

            def wrap_socket(self, sock: FakeSock, server_hostname: str | None = None) -> FakeSock:
                return FakeSock()

        def fake_create_connection(address, timeout=None):  # noqa: ANN001
            calls.append(address)
            return FakeSock()

        with mock.patch("socket.getaddrinfo", return_value=infos):
            target = api_pull.validate_pull_url(
                "https://stable.example/",
                allow_hosts=["stable.example"],
                allow_private=False,
            )

        evil_infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("9.9.9.9", 443))]
        with (
            mock.patch("socket.getaddrinfo", return_value=evil_infos) as gai,
            mock.patch("socket.create_connection", side_effect=fake_create_connection),
            mock.patch("ssl.create_default_context", return_value=FakeContext()),
        ):
            api_pull._open_connection(target, timeout=2.0)
            gai.assert_not_called()
        self.assertEqual(calls, [("8.8.4.4", 443)])


if __name__ == "__main__":
    unittest.main()
