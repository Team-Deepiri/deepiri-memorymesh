"""Batch 6: T37 project-scoped HTTP bearer-token authentication/authorization.

Covers: token format, absence from DB, create/list/revoke/rotate, expiration,
scopes, project isolation, revoked/malformed/missing tokens, ``/health`` open,
``auth_mode=off`` warning, concurrent verification, no-token-in-errors, and
bridge/OpenCode-plugin source containing only env/file references (never a
literal token).
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib import error as urlerror
from urllib import request as urlrequest

from typer.testing import CliRunner

from tests.helpers import temp_home

from deepiri_memorymesh import auth
from deepiri_memorymesh import service_api
from deepiri_memorymesh.cli import app
from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.integrations import (
    _curl_ingest_from_env_fragment,
    _opencode_plugin_source,
    install_bridge_script,
    install_hook_script,
)
from deepiri_memorymesh.storage import MemoryStore
from deepiri_memorymesh.sync_service import MemoryMesh
from deepiri_memorymesh.supervised_service import SupervisedService


def _http_get(base_url: str, path: str, *, token: str | None = None):
    req = urlrequest.Request(base_url + path, method="GET")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urlrequest.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, (json.loads(body) if body else {})


def _http_post(
    base_url: str,
    path: str,
    payload: dict,
    *,
    token: str | None = None,
    raw_auth_header: str | None = None,
):
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(base_url + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if raw_auth_header is not None:
        req.add_header("Authorization", raw_auth_header)
    elif token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urlrequest.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, (json.loads(body) if body else {})


class TokenFormatTests(unittest.TestCase):
    def test_format_matches_spec(self) -> None:
        token_id = auth._new_token_id()
        secret = auth._new_secret()
        self.assertEqual(len(token_id), 16)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{16}", token_id))
        # 32 bytes urlsafe-base64 -> 43 chars (no padding).
        self.assertEqual(len(secret), 43)
        self.assertTrue(re.fullmatch(r"[A-Za-z0-9_-]+", secret))

        token = auth.format_token(token_id, secret)
        self.assertEqual(token, f"mmht.{token_id}.{secret}")
        self.assertTrue(token.startswith("mmht."))

    def test_parse_token_round_trip(self) -> None:
        token_id = auth._new_token_id()
        secret = auth._new_secret()
        token = auth.format_token(token_id, secret)
        parsed = auth.parse_token(token)
        self.assertEqual(parsed, (token_id, secret))

    def test_parse_token_rejects_malformed(self) -> None:
        for bad in [
            "",
            "not-a-token",
            "mmht.short.secretsecretsecretsecretsecretsecret",
            "mmht.ZZZZZZZZZZZZZZZZ.secretsecretsecretsecretsecretsecretsec",  # non-hex id
            "mmht." + "a" * 16 + ".",  # empty secret
            "MMHT." + "a" * 16 + "." + "b" * 40,  # wrong case prefix
            "mmht" + "a" * 16 + "b" * 40,  # missing dots
            None,  # type: ignore[arg-type]
            123,  # type: ignore[arg-type]
        ]:
            self.assertIsNone(auth.parse_token(bad))  # type: ignore[arg-type]


class TokenLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.db_path = self.root / "mesh.db"
        MemoryStore(self.db_path).init()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_create_token_stores_only_hash_never_plaintext(self) -> None:
        plaintext, record = auth.create_token(self.db_path, "proj1", ["read", "write"], label="ci")
        self.assertTrue(plaintext.startswith("mmht."))
        self.assertEqual(record.project, "proj1")
        self.assertEqual(record.scopes, ("read", "write"))
        self.assertEqual(record.label, "ci")
        self.assertIsNone(record.revoked_at)

        raw_db_bytes = self.db_path.read_bytes()
        self.assertNotIn(plaintext.encode("utf-8"), raw_db_bytes)
        parsed = auth.parse_token(plaintext)
        assert parsed is not None
        _token_id, secret = parsed
        self.assertNotIn(secret.encode("utf-8"), raw_db_bytes)
        self.assertIn(record.token_id.encode("utf-8"), raw_db_bytes)

    def test_absent_token_returns_401(self) -> None:
        # Well-formed token, but never created / not in the DB.
        never_created = auth.format_token(auth._new_token_id(), auth._new_secret())
        result = auth.verify_bearer(
            self.db_path, f"Bearer {never_created}", project="proj1", required_scope="read"
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.status, 401)
        self.assertEqual(result.error, "invalid_token")

    def test_verify_bearer_on_uninitialized_db_fails_closed(self) -> None:
        missing_db = self.root / "does-not-exist.db"
        token = auth.format_token(auth._new_token_id(), auth._new_secret())
        result = auth.verify_bearer(
            missing_db, f"Bearer {token}", project="proj1", required_scope="read"
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.status, 401)

    def test_list_tokens_never_includes_secrets(self) -> None:
        plaintext, record = auth.create_token(self.db_path, "proj1", ["read"])
        records = auth.list_tokens(self.db_path, "proj1")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].token_id, record.token_id)
        dump = json.dumps([r.__dict__ if hasattr(r, "__dict__") else str(r) for r in records])
        self.assertNotIn(plaintext, dump)
        self.assertNotIn(plaintext.split(".")[-1], dump)

    def test_list_tokens_scoped_by_project(self) -> None:
        auth.create_token(self.db_path, "proj1", ["read"])
        auth.create_token(self.db_path, "proj2", ["read"])
        only_proj1 = auth.list_tokens(self.db_path, "proj1")
        self.assertEqual(len(only_proj1), 1)
        self.assertEqual(only_proj1[0].project, "proj1")
        everything = auth.list_tokens(self.db_path)
        self.assertEqual(len(everything), 2)

    def test_revoke_token_then_verify_fails(self) -> None:
        plaintext, record = auth.create_token(self.db_path, "proj1", ["read"])
        revoked = auth.revoke_token(self.db_path, record.token_id)
        self.assertIsNotNone(revoked.revoked_at)

        result = auth.verify_bearer(
            self.db_path, f"Bearer {plaintext}", project="proj1", required_scope="read"
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.status, 401)

    def test_revoke_is_idempotent(self) -> None:
        _plaintext, record = auth.create_token(self.db_path, "proj1", ["read"])
        first = auth.revoke_token(self.db_path, record.token_id)
        second = auth.revoke_token(self.db_path, record.token_id)
        self.assertEqual(first.revoked_at, second.revoked_at)

    def test_revoke_unknown_token_raises(self) -> None:
        with self.assertRaises(auth.TokenNotFoundError):
            auth.revoke_token(self.db_path, "0" * 16)

    def test_rotate_token_creates_new_and_revokes_old(self) -> None:
        old_plaintext, old_record = auth.create_token(
            self.db_path, "proj1", ["read", "write"], label="rotating"
        )
        new_plaintext, new_record = auth.rotate_token(self.db_path, old_record.token_id)

        self.assertNotEqual(new_plaintext, old_plaintext)
        self.assertNotEqual(new_record.token_id, old_record.token_id)
        self.assertEqual(new_record.project, old_record.project)
        self.assertEqual(new_record.scopes, old_record.scopes)
        self.assertEqual(new_record.label, old_record.label)

        old_check = auth.verify_bearer(
            self.db_path, f"Bearer {old_plaintext}", project="proj1", required_scope="read"
        )
        self.assertFalse(old_check.ok)

        new_check = auth.verify_bearer(
            self.db_path, f"Bearer {new_plaintext}", project="proj1", required_scope="read"
        )
        self.assertTrue(new_check.ok)

    def test_rotate_unknown_token_raises(self) -> None:
        with self.assertRaises(auth.TokenNotFoundError):
            auth.rotate_token(self.db_path, "1" * 16)

    def test_expired_token_rejected(self) -> None:
        past = "2000-01-01T00:00:00+00:00"
        plaintext, _record = auth.create_token(
            self.db_path, "proj1", ["read"], expires_at=past
        )
        result = auth.verify_bearer(
            self.db_path, f"Bearer {plaintext}", project="proj1", required_scope="read"
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.status, 401)

    def test_future_expiry_still_valid(self) -> None:
        future = "2099-01-01T00:00:00+00:00"
        plaintext, _record = auth.create_token(
            self.db_path, "proj1", ["read"], expires_at=future
        )
        result = auth.verify_bearer(
            self.db_path, f"Bearer {plaintext}", project="proj1", required_scope="read"
        )
        self.assertTrue(result.ok)

    def test_scopes_enforced_read_vs_write(self) -> None:
        read_only, _ = auth.create_token(self.db_path, "proj1", ["read"])
        write_only, _ = auth.create_token(self.db_path, "proj1", ["write"])

        r1 = auth.verify_bearer(self.db_path, f"Bearer {read_only}", project="proj1", required_scope="read")
        self.assertTrue(r1.ok)
        r2 = auth.verify_bearer(self.db_path, f"Bearer {read_only}", project="proj1", required_scope="write")
        self.assertFalse(r2.ok)
        self.assertEqual(r2.status, 403)
        self.assertEqual(r2.error, "insufficient_scope")

        r3 = auth.verify_bearer(self.db_path, f"Bearer {write_only}", project="proj1", required_scope="write")
        self.assertTrue(r3.ok)
        r4 = auth.verify_bearer(self.db_path, f"Bearer {write_only}", project="proj1", required_scope="read")
        self.assertFalse(r4.ok)
        self.assertEqual(r4.status, 403)

    def test_invalid_scope_rejected_at_creation(self) -> None:
        with self.assertRaises(auth.InvalidScopeError):
            auth.create_token(self.db_path, "proj1", ["admin"])
        with self.assertRaises(auth.InvalidScopeError):
            auth.create_token(self.db_path, "proj1", [])

    def test_project_isolation_returns_403(self) -> None:
        plaintext, _ = auth.create_token(self.db_path, "proj-a", ["read", "write"])
        result = auth.verify_bearer(
            self.db_path, f"Bearer {plaintext}", project="proj-b", required_scope="read"
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.status, 403)
        self.assertEqual(result.error, "wrong_project")

    def test_malformed_authorization_header_rejected(self) -> None:
        for header in [
            "Bearer",
            "Bearer ",
            "Basic dXNlcjpwYXNz",
            "mmht.abc.def",  # missing "Bearer " scheme
            "Bearer not-a-real-token-format",
            "Bearer mmht.tooshort.def",
            "Bearer " + "x" * 5000,  # absurdly long garbage
        ]:
            result = auth.verify_bearer(
                self.db_path, header, project="proj1", required_scope="read"
            )
            self.assertFalse(result.ok, msg=header)
            self.assertEqual(result.status, 401, msg=header)

    def test_missing_authorization_header_rejected(self) -> None:
        for header in [None, "", "   "]:
            result = auth.verify_bearer(
                self.db_path, header, project="proj1", required_scope="read"
            )
            self.assertFalse(result.ok)
            self.assertEqual(result.status, 401)
            self.assertEqual(result.error, "missing_token")

    def test_no_presented_token_in_verification_result(self) -> None:
        plaintext, record = auth.create_token(self.db_path, "proj1", ["read"])
        # Tamper the secret so it fails verification, and ensure the tampered
        # (and original) token text never leaks into the failure result.
        tampered = plaintext[:-1] + ("A" if plaintext[-1] != "A" else "B")
        result = auth.verify_bearer(
            self.db_path, f"Bearer {tampered}", project="proj1", required_scope="read"
        )
        self.assertFalse(result.ok)
        dump = json.dumps(
            {"error": result.error, "message": result.message, "token_id": result.token_id}
        )
        self.assertNotIn(tampered, dump)
        self.assertNotIn(plaintext, dump)
        self.assertNotIn(plaintext.split(".")[-1], dump)  # secret

    def test_no_token_in_exceptions(self) -> None:
        secret_marker = "s3cr3t-should-never-appear-in-any-exception"
        crafted = f"mmht.{'a' * 16}.{secret_marker}-------------------------"
        try:
            auth.verify_bearer(self.db_path, f"Bearer {crafted}", project="proj1", required_scope="read")
        except Exception as exc:  # pragma: no cover - verify_bearer should not raise
            self.fail(f"verify_bearer raised unexpectedly: {exc!r}")
        # Also exercise revoke/rotate NotFound exceptions for token-text leakage.
        with self.assertRaises(auth.TokenNotFoundError) as ctx:
            auth.revoke_token(self.db_path, "deadbeefdeadbeef")
        self.assertNotIn(secret_marker, str(ctx.exception))

    def test_write_token_file_mode_0600(self) -> None:
        plaintext, _record = auth.create_token(self.db_path, "proj1", ["read"])
        token_file = self.root / "token.txt"
        auth.write_token_file(token_file, plaintext)
        mode = token_file.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)
        self.assertEqual(token_file.read_text(encoding="utf-8").strip(), plaintext)

    def test_write_token_file_refuses_overwrite_by_default(self) -> None:
        _plaintext, _record = auth.create_token(self.db_path, "proj1", ["read"])
        token_file = self.root / "token.txt"
        auth.write_token_file(token_file, "first")
        with self.assertRaises(auth.AuthError):
            auth.write_token_file(token_file, "second")
        auth.write_token_file(token_file, "second", overwrite=True)
        self.assertEqual(token_file.read_text(encoding="utf-8").strip(), "second")

    def test_auth_status_reports_counts_without_secrets(self) -> None:
        plaintext, record = auth.create_token(self.db_path, "proj1", ["read"])
        auth.revoke_token(self.db_path, record.token_id)
        auth.create_token(self.db_path, "proj2", ["write"])

        status = auth.auth_status(self.db_path)
        self.assertTrue(status.schema_ready)
        self.assertEqual(status.total_tokens, 2)
        self.assertEqual(status.active_tokens, 1)
        self.assertEqual(status.revoked_tokens, 1)
        self.assertIn("proj1", status.projects)
        self.assertIn("proj2", status.projects)
        self.assertNotIn(plaintext, repr(status))

    def test_concurrent_verify_same_token(self) -> None:
        plaintext, record = auth.create_token(self.db_path, "proj1", ["read"])
        results: list[bool] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                res = auth.verify_bearer(
                    self.db_path, f"Bearer {plaintext}", project="proj1", required_scope="read"
                )
                with lock:
                    results.append(res.ok)
            except BaseException as exc:  # pragma: no cover
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 12)
        self.assertTrue(all(results))

        # last_used_at should have been updated (best-effort) by at least one call.
        records = auth.list_tokens(self.db_path, "proj1")
        self.assertEqual(len(records), 1)
        self.assertIsNotNone(records[0].last_used_at)


class ServiceApiAuthIntegrationTests(unittest.TestCase):
    """Live loopback-HTTP integration coverage for T37 (real request round-trips)."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.home = self.root / "home"
        self._home_ctx = temp_home(self.home)
        self._home_ctx.__enter__()
        self.db_path = self.root / "svc.db"
        MemoryStore(self.db_path).init()
        self.settings = Settings(db_path=self.db_path, embedding_backend="fallback")

        probe = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        self.port = probe.server_address[1]
        probe.server_close()
        self.svc: SupervisedService | None = None

    def tearDown(self) -> None:
        if self.svc is not None:
            self.svc.shutdown()
        self._home_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def _start(self, auth_mode: str = "required") -> str:
        self.svc = SupervisedService(
            host="127.0.0.1", port=self.port, settings=self.settings, auth_mode=auth_mode
        )
        result = self.svc.start()
        self.assertEqual(result, "started")
        return f"http://127.0.0.1:{self.port}"

    def _ingest_payload(self, project: str) -> dict:
        return {
            "provider": "cursor",
            "project": project,
            "conversation": {
                "conversation_id": "c1",
                "messages": [{"role": "user", "content": "hello there"}],
            },
        }

    def test_health_open_without_token(self) -> None:
        base = self._start(auth_mode="required")
        status, body = _http_get(base, "/health")
        self.assertEqual(status, 200)
        self.assertTrue(body.get("ok"))

    def test_ingest_requires_write_scope_for_exact_project(self) -> None:
        base = self._start(auth_mode="required")
        payload = self._ingest_payload("proj1")

        status, body = _http_post(base, "/ingest", payload)
        self.assertEqual(status, 401)
        self.assertNotIn("mmht.", json.dumps(body))

        read_only, _ = auth.create_token(self.db_path, "proj1", ["read"])
        status, _body = _http_post(base, "/ingest", payload, token=read_only)
        self.assertEqual(status, 403)

        write_token, _ = auth.create_token(self.db_path, "proj1", ["write"])
        status, body = _http_post(base, "/ingest", payload, token=write_token)
        self.assertEqual(status, 200)
        self.assertTrue(body.get("ok"))

    def test_query_requires_read_scope_for_exact_project(self) -> None:
        base = self._start(auth_mode="required")
        write_token, _ = auth.create_token(self.db_path, "proj1", ["write"])
        status, _ = _http_post(base, "/ingest", self._ingest_payload("proj1"), token=write_token)
        self.assertEqual(status, 200)

        query_payload = {"project": "proj1", "q": "hello"}
        status, _ = _http_post(base, "/query", query_payload)
        self.assertEqual(status, 401)

        status, _ = _http_post(base, "/query", query_payload, token=write_token)
        self.assertEqual(status, 403)

        read_token, _ = auth.create_token(self.db_path, "proj1", ["read"])
        status, body = _http_post(base, "/query", query_payload, token=read_token)
        self.assertEqual(status, 200)
        self.assertTrue(body.get("ok"))

    def test_state_put_get_scopes(self) -> None:
        base = self._start(auth_mode="required")
        write_token, _ = auth.create_token(self.db_path, "proj1", ["write"])
        read_token, _ = auth.create_token(self.db_path, "proj1", ["read"])

        put_payload = {"project": "proj1", "agent": "a1", "key": "k", "value": "v"}
        status, _ = _http_post(base, "/state/put", put_payload)
        self.assertEqual(status, 401)
        status, _ = _http_post(base, "/state/put", put_payload, token=read_token)
        self.assertEqual(status, 403)
        status, _ = _http_post(base, "/state/put", put_payload, token=write_token)
        self.assertEqual(status, 200)

        get_payload = {"project": "proj1", "agent": "a1", "key": "k"}
        status, _ = _http_post(base, "/state/get", get_payload)
        self.assertEqual(status, 401)
        status, _ = _http_post(base, "/state/get", get_payload, token=write_token)
        self.assertEqual(status, 403)
        status, body = _http_post(base, "/state/get", get_payload, token=read_token)
        self.assertEqual(status, 200)
        self.assertEqual(body.get("value"), "v")

    def test_stats_requires_read_scope(self) -> None:
        base = self._start(auth_mode="required")
        status, _ = _http_get(base, "/stats?project=proj1")
        self.assertEqual(status, 401)

        write_token, _ = auth.create_token(self.db_path, "proj1", ["write"])
        status, _ = _http_get(base, "/stats?project=proj1", token=write_token)
        self.assertEqual(status, 403)

        read_token, _ = auth.create_token(self.db_path, "proj1", ["read"])
        status, body = _http_get(base, "/stats?project=proj1", token=read_token)
        self.assertEqual(status, 200)
        self.assertTrue(body.get("ok"))

    def test_wrong_project_returns_403(self) -> None:
        base = self._start(auth_mode="required")
        token, _ = auth.create_token(self.db_path, "proj-a", ["read", "write"])
        status, _ = _http_post(base, "/ingest", self._ingest_payload("proj-b"), token=token)
        self.assertEqual(status, 403)

    def test_revoked_token_rejected_over_http(self) -> None:
        base = self._start(auth_mode="required")
        token, record = auth.create_token(self.db_path, "proj1", ["read", "write"])
        auth.revoke_token(self.db_path, record.token_id)
        status, _ = _http_post(base, "/ingest", self._ingest_payload("proj1"), token=token)
        self.assertEqual(status, 401)

    def test_malformed_bearer_rejected_over_http(self) -> None:
        base = self._start(auth_mode="required")
        for raw_header in ["Bearer garbage", "Bearer ", "Basic abc123", "totally-invalid"]:
            status, _ = _http_post(
                base,
                "/ingest",
                self._ingest_payload("proj1"),
                raw_auth_header=raw_header,
            )
            self.assertEqual(status, 401, msg=raw_header)

    def test_query_string_token_is_ignored(self) -> None:
        base = self._start(auth_mode="required")
        write_token, _ = auth.create_token(self.db_path, "proj1", ["write"])
        # Putting the token in the query string must NOT authenticate the
        # request; the server never even routes it as /ingest with a token
        # attached, and no Authorization header means no ingest happens.
        status, body = _http_post(
            base, f"/ingest?token={write_token}", self._ingest_payload("proj1")
        )
        self.assertNotEqual(status, 200)
        self.assertNotIn(write_token, json.dumps(body))

    def test_auth_off_allows_requests_without_token(self) -> None:
        base = self._start(auth_mode="off")
        status, body = _http_post(base, "/ingest", self._ingest_payload("proj1"))
        self.assertEqual(status, 200)
        self.assertTrue(body.get("ok"))

    def test_no_token_leaked_in_http_error_bodies(self) -> None:
        base = self._start(auth_mode="required")
        secret_marker = "s3cr3t-must-not-leak-anywhere"
        crafted_header = f"Bearer mmht.{'a' * 16}.{secret_marker}" + "-" * 20
        status, body = _http_post(
            base, "/ingest", self._ingest_payload("proj1"), raw_auth_header=crafted_header
        )
        self.assertEqual(status, 401)
        self.assertNotIn(secret_marker, json.dumps(body))


class AuthOffWarningTests(unittest.TestCase):
    """``run_service(auth_mode="off")`` must print a visible warning."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.home = self.root / "home"
        self._home_ctx = temp_home(self.home)
        self._home_ctx.__enter__()
        self.settings = Settings(db_path=self.root / "svc.db", embedding_backend="fallback")

    def tearDown(self) -> None:
        self._home_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def _run_with_fake_server(self, auth_mode: str):
        class _FakeServer:
            def serve_forever(self_inner) -> None:  # noqa: N805
                return None

        fake = _FakeServer()
        buf = io.StringIO()
        import unittest.mock as mock

        with mock.patch.object(service_api, "make_http_server", return_value=fake):
            with contextlib.redirect_stdout(buf):
                service_api.run_service(
                    host="127.0.0.1", port=0, settings=self.settings, auth_mode=auth_mode
                )
        return buf.getvalue(), fake

    def test_auth_off_prints_warning(self) -> None:
        output, fake = self._run_with_fake_server("off")
        self.assertIn("WARNING", output)
        self.assertIn("DISABLED", output)
        self.assertIn("auth_mode=off", output)

    def test_auth_required_does_not_print_warning(self) -> None:
        output, _fake = self._run_with_fake_server("required")
        self.assertNotIn("WARNING", output)
        self.assertIn("auth_mode=required", output)

    def test_invalid_auth_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            service_api.run_service(
                host="127.0.0.1", port=0, settings=self.settings, auth_mode="bogus"
            )


class BridgeAndPluginSourceTests(unittest.TestCase):
    """Generated bridge/hook/plugin sources must reference env vars, never literal tokens."""

    def test_curl_fragment_references_env_vars_only(self) -> None:
        frag = _curl_ingest_from_env_fragment()
        self.assertIn("MEMORYMESH_TOKEN", frag)
        self.assertIn("MEMORYMESH_TOKEN_FILE", frag)
        self.assertIn("Authorization: Bearer", frag)
        # No plausible literal token embedded (format: mmht.<16hex>.<secret>).
        self.assertNotRegex(frag, r"mmht\.[0-9a-f]{16}\.[A-Za-z0-9_-]{16,}")

    def test_opencode_plugin_source_references_env_vars_only(self) -> None:
        src = _opencode_plugin_source("proj1", "http://127.0.0.1:8765")
        self.assertIn("MEMORYMESH_TOKEN", src)
        self.assertIn("MEMORYMESH_TOKEN_FILE", src)
        self.assertIn("Authorization", src)
        self.assertNotRegex(src, r"mmht\.[0-9a-f]{16}\.[A-Za-z0-9_-]{16,}")

    def test_installed_bridge_and_hook_scripts_contain_env_refs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / "home"
            with temp_home(home):
                bridge = install_bridge_script("cursor", "proj1", "http://127.0.0.1:8765")
                hook = install_hook_script("cursor", "proj1", "http://127.0.0.1:8765")
                for script in (bridge, hook):
                    text = script.read_text(encoding="utf-8")
                    self.assertIn("MEMORYMESH_TOKEN", text)
                    self.assertIn("MEMORYMESH_TOKEN_FILE", text)
                    self.assertNotRegex(text, r"mmht\.[0-9a-f]{16}\.[A-Za-z0-9_-]{16,}")


class CliAuthCommandsTests(unittest.TestCase):
    """``memorymesh auth ...`` and ``serve --auth-mode`` CLI wiring."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.db_path = self.root / "cli.db"
        self.settings = Settings(db_path=self.db_path, embedding_backend="fallback")
        self.mesh = MemoryMesh(self.settings)
        self.mesh.init()
        self.runner = CliRunner()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _invoke(self, args: list[str]):
        with mock.patch("deepiri_memorymesh.cli._mesh", return_value=self.mesh):
            return self.runner.invoke(app, args)

    @staticmethod
    def _combined(result) -> str:
        return (result.stdout or "") + (getattr(result, "stderr", "") or "")

    def test_create_without_scope_fails_clearly(self) -> None:
        result = self._invoke(["auth", "token", "create", "--project", "proj1"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--scope", self._combined(result))

    def test_create_list_revoke_rotate_flow(self) -> None:
        create = self._invoke(
            [
                "auth", "token", "create",
                "--project", "proj1",
                "--scope", "read",
                "--scope", "write",
                "--label", "ci",
            ]
        )
        self.assertEqual(create.exit_code, 0, self._combined(create))
        out_lines = create.stdout.splitlines()
        plaintext_lines = [l for l in out_lines if l.startswith("mmht.")]
        self.assertEqual(len(plaintext_lines), 1)
        plaintext = plaintext_lines[0]
        token_id = None
        for line in out_lines:
            if line.startswith("token_id="):
                token_id = line.split()[0].split("=", 1)[1]
        self.assertIsNotNone(token_id)

        listing = self._invoke(["auth", "token", "list", "--project", "proj1"])
        self.assertEqual(listing.exit_code, 0, self._combined(listing))
        self.assertIn(token_id, listing.stdout)
        self.assertNotIn(plaintext, listing.stdout)
        self.assertNotIn(plaintext.split(".")[-1], listing.stdout)

        revoke = self._invoke(["auth", "token", "revoke", "--id", token_id])
        self.assertEqual(revoke.exit_code, 0, self._combined(revoke))
        self.assertIn("revoked", revoke.stdout)
        self.assertIn(token_id, revoke.stdout)

        rotate = self._invoke(["auth", "token", "rotate", "--id", token_id])
        self.assertEqual(rotate.exit_code, 0, self._combined(rotate))
        rotate_lines = [l for l in rotate.stdout.splitlines() if l.startswith("mmht.")]
        self.assertEqual(len(rotate_lines), 1)
        self.assertNotEqual(rotate_lines[0], plaintext)

    def test_revoke_unknown_id_fails_clearly(self) -> None:
        result = self._invoke(["auth", "token", "revoke", "--id", "0" * 16])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No token found", self._combined(result))

    def test_status_reports_counts(self) -> None:
        self._invoke(["auth", "token", "create", "--project", "p1", "--scope", "read"])
        status = self._invoke(["auth", "status"])
        self.assertEqual(status.exit_code, 0, self._combined(status))
        self.assertIn("total_tokens=1", status.stdout)
        self.assertIn("active_tokens=1", status.stdout)

    def test_token_file_written_with_mode_0600(self) -> None:
        token_file = self.root / "token.txt"
        result = self._invoke(
            [
                "auth", "token", "create",
                "--project", "proj1",
                "--scope", "read",
                "--token-file", str(token_file),
            ]
        )
        self.assertEqual(result.exit_code, 0, self._combined(result))
        self.assertTrue(token_file.exists())
        mode = token_file.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)
        content = token_file.read_text(encoding="utf-8").strip()
        self.assertTrue(content.startswith("mmht."))

    def test_serve_auth_mode_off_warns_and_forwards_flag(self) -> None:
        home = self.root / "serve-home"
        with temp_home(home):
            with mock.patch("deepiri_memorymesh.cli.run_service") as run_service_mock:
                result = self.runner.invoke(app, ["serve", "--auth-mode", "off"])
        self.assertEqual(result.exit_code, 0, self._combined(result))
        self.assertIn("WARNING", self._combined(result))
        run_service_mock.assert_called_once()
        _args, kwargs = run_service_mock.call_args
        self.assertEqual(kwargs.get("auth_mode"), "off")

    def test_serve_default_auth_mode_required_no_warning(self) -> None:
        home = self.root / "serve-home2"
        with temp_home(home):
            with mock.patch("deepiri_memorymesh.cli.run_service") as run_service_mock:
                result = self.runner.invoke(app, ["serve"])
        self.assertEqual(result.exit_code, 0, self._combined(result))
        self.assertNotIn("WARNING", self._combined(result))
        run_service_mock.assert_called_once()
        _args, kwargs = run_service_mock.call_args
        self.assertEqual(kwargs.get("auth_mode"), "required")

    def test_serve_rejects_invalid_auth_mode(self) -> None:
        home = self.root / "serve-home3"
        with temp_home(home):
            with mock.patch("deepiri_memorymesh.cli.run_service") as run_service_mock:
                result = self.runner.invoke(app, ["serve", "--auth-mode", "bogus"])
        self.assertNotEqual(result.exit_code, 0)
        run_service_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
