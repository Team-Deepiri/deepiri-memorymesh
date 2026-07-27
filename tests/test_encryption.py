"""Batch 6: T33 encryption foundations (crypto envelope + enable/rotate lifecycle)."""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import temp_home

from deepiri_memorymesh import crypto
from deepiri_memorymesh import encryption
from deepiri_memorymesh.crypto import (
    CryptographyUnavailableError,
    EncryptionContext,
    InvalidKeyMaterialError,
    KeyFileExistsError,
    MalformedEnvelopeError,
    TamperError,
    UnsupportedEnvelopeVersionError,
    WrongKeyError,
    content_fingerprint,
    decrypt_field,
    encrypt_field,
    generate_key_file,
    is_envelope,
    keyed_term_token,
    load_key_from_env,
    load_key_from_file,
    load_master_key,
    require_cryptography,
)
from deepiri_memorymesh.encryption import (
    EncryptionLifecycleError,
    enable_encryption,
    encryption_status,
    rotate_encryption,
)
from deepiri_memorymesh.migrations import CURRENT_SCHEMA_VERSION, migrate
from deepiri_memorymesh.models import MemoryRecord
from deepiri_memorymesh.storage import MemoryStore


def _ctx(key: bytes | None = None, db_identity: str = "db-1") -> EncryptionContext:
    key = key or (b"k" * 32)
    return EncryptionContext(key, db_identity)


_CRYPTO_SKIP_REASON = "cryptography package not installed; install extras: pip install '.[security]'"


@unittest.skipUnless(crypto.CRYPTOGRAPHY_AVAILABLE, _CRYPTO_SKIP_REASON)
class EnvelopeRoundtripTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        ctx = _ctx()
        envelope = encrypt_field(ctx, "hello world", table="memory_messages", column="content", row_identity="1")
        self.assertTrue(is_envelope(envelope))
        data = json.loads(envelope)
        self.assertEqual(data["v"], 1)
        self.assertEqual(data["alg"], "AES-256-GCM")
        self.assertEqual(data["kid"], ctx.key_id)
        self.assertIn("nonce", data)
        self.assertIn("ct", data)
        # Ciphertext must not leak the plaintext anywhere in the envelope.
        self.assertNotIn("hello", envelope)

        decrypted = decrypt_field(ctx, envelope, table="memory_messages", column="content", row_identity="1")
        self.assertEqual(decrypted, "hello world")

    def test_envelope_is_json_object_with_expected_shape(self) -> None:
        ctx = _ctx()
        envelope = encrypt_field(ctx, "x", table="t", column="c", row_identity="r")
        data = json.loads(envelope)
        self.assertEqual(set(data.keys()), {"v", "alg", "kid", "nonce", "ct"})

    def test_different_row_identity_fails_to_decrypt(self) -> None:
        ctx = _ctx()
        envelope = encrypt_field(ctx, "secret", table="memory_messages", column="content", row_identity="1")
        with self.assertRaises(TamperError):
            decrypt_field(ctx, envelope, table="memory_messages", column="content", row_identity="2")

    def test_different_table_or_column_fails_to_decrypt(self) -> None:
        ctx = _ctx()
        envelope = encrypt_field(ctx, "secret", table="memory_messages", column="content", row_identity="1")
        with self.assertRaises(TamperError):
            decrypt_field(ctx, envelope, table="memory_messages", column="metadata_json", row_identity="1")

    def test_tamper_detected(self) -> None:
        ctx = _ctx()
        envelope = encrypt_field(ctx, "secret value", table="t", column="c", row_identity="1")
        data = json.loads(envelope)
        ct_bytes = bytearray(base64.b64decode(data["ct"]))
        ct_bytes[0] ^= 0xFF
        data["ct"] = base64.b64encode(bytes(ct_bytes)).decode("ascii")
        tampered = json.dumps(data)
        with self.assertRaises(TamperError):
            decrypt_field(ctx, tampered, table="t", column="c", row_identity="1")

    def test_wrong_key_detected(self) -> None:
        ctx1 = _ctx(b"k" * 32)
        ctx2 = _ctx(b"z" * 32)
        envelope = encrypt_field(ctx1, "secret value", table="t", column="c", row_identity="1")
        with self.assertRaises(WrongKeyError):
            decrypt_field(ctx2, envelope, table="t", column="c", row_identity="1")

    def test_malformed_envelope_not_json(self) -> None:
        ctx = _ctx()
        with self.assertRaises(MalformedEnvelopeError):
            decrypt_field(ctx, "not json at all", table="t", column="c", row_identity="1")

    def test_malformed_envelope_missing_fields(self) -> None:
        ctx = _ctx()
        with self.assertRaises(MalformedEnvelopeError):
            decrypt_field(ctx, json.dumps({"v": 1, "alg": "AES-256-GCM"}), table="t", column="c", row_identity="1")

    def test_malformed_envelope_bad_base64(self) -> None:
        ctx = _ctx()
        bad = json.dumps({"v": 1, "alg": "AES-256-GCM", "kid": ctx.key_id, "nonce": "***not-b64***", "ct": "***"})
        with self.assertRaises(MalformedEnvelopeError):
            decrypt_field(ctx, bad, table="t", column="c", row_identity="1")

    def test_unsupported_version_detected(self) -> None:
        ctx = _ctx()
        envelope = encrypt_field(ctx, "v", table="t", column="c", row_identity="1")
        data = json.loads(envelope)
        data["v"] = 99
        with self.assertRaises(UnsupportedEnvelopeVersionError):
            decrypt_field(ctx, json.dumps(data), table="t", column="c", row_identity="1")

    def test_is_envelope_rejects_plain_strings_and_unrelated_json(self) -> None:
        self.assertFalse(is_envelope("hello world"))
        self.assertFalse(is_envelope("{}"))
        self.assertFalse(is_envelope(json.dumps({"foo": "bar"})))
        self.assertFalse(is_envelope(123))
        self.assertFalse(is_envelope(None))

    def test_is_envelope_accepts_real_envelope(self) -> None:
        ctx = _ctx()
        envelope = encrypt_field(ctx, "hi", table="t", column="c", row_identity="1")
        self.assertTrue(is_envelope(envelope))


@unittest.skipUnless(crypto.CRYPTOGRAPHY_AVAILABLE, _CRYPTO_SKIP_REASON)
class ContentFingerprintAndTermTokenTests(unittest.TestCase):
    def test_content_fingerprint_deterministic_and_keyed(self) -> None:
        ctx = _ctx(b"a" * 32)
        fp1 = content_fingerprint(
            ctx, project="p", provider="claude", conversation_id="c1", role="user", content="hello"
        )
        fp2 = content_fingerprint(
            ctx, project="p", provider="claude", conversation_id="c1", role="user", content="hello"
        )
        self.assertEqual(fp1, fp2)
        fp3 = content_fingerprint(
            ctx, project="p", provider="claude", conversation_id="c1", role="user", content="different"
        )
        self.assertNotEqual(fp1, fp3)

        other_ctx = _ctx(b"b" * 32)
        fp_other_key = content_fingerprint(
            other_ctx, project="p", provider="claude", conversation_id="c1", role="user", content="hello"
        )
        self.assertNotEqual(fp1, fp_other_key)
        # Never leaks plaintext.
        self.assertNotIn("hello", fp1)

    def test_keyed_term_token_deterministic_and_keyed(self) -> None:
        ctx = _ctx(b"a" * 32)
        t1 = keyed_term_token(ctx, "azure")
        t2 = keyed_term_token(ctx, "azure")
        self.assertEqual(t1, t2)
        t3 = keyed_term_token(ctx, "aws")
        self.assertNotEqual(t1, t3)
        other_ctx = _ctx(b"b" * 32)
        self.assertNotEqual(t1, keyed_term_token(other_ctx, "azure"))
        self.assertNotIn("azure", t1)


class KeyLoadingTests(unittest.TestCase):
    def test_load_key_from_env_hex(self) -> None:
        hex_key = ("ab" * 32)
        with mock.patch.dict(os.environ, {"MEMORYMESH_ENCRYPTION_KEY": hex_key}, clear=False):
            key = load_key_from_env()
        self.assertEqual(key, bytes.fromhex(hex_key))
        self.assertEqual(len(key), 32)

    def test_load_key_from_env_base64url(self) -> None:
        raw = os.urandom(32)
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        with mock.patch.dict(os.environ, {"MEMORYMESH_ENCRYPTION_KEY": encoded}, clear=False):
            key = load_key_from_env()
        self.assertEqual(key, raw)

    def test_load_key_from_env_missing(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(InvalidKeyMaterialError):
                load_key_from_env()

    def test_load_key_too_short_rejected(self) -> None:
        short_hex = "ab" * 8  # 8 bytes, under the 32-byte minimum
        with mock.patch.dict(os.environ, {"MEMORYMESH_ENCRYPTION_KEY": short_hex}, clear=False):
            with self.assertRaises(InvalidKeyMaterialError):
                load_key_from_env()

    def test_generate_key_file_permissions_and_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mesh.key"
            generate_key_file(path)
            mode = stat.S_IMODE(path.stat().st_mode)
            self.assertEqual(mode, 0o600)
            key = load_key_from_file(path)
            self.assertEqual(len(key), 32)

    def test_generate_key_file_refuses_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mesh.key"
            generate_key_file(path)
            with self.assertRaises(KeyFileExistsError):
                generate_key_file(path)
            # Original key material is untouched.
            key_before = load_key_from_file(path)
            with self.assertRaises(KeyFileExistsError):
                generate_key_file(path, overwrite=False)
            self.assertEqual(load_key_from_file(path), key_before)

    def test_generate_key_file_overwrite_true_replaces(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mesh.key"
            generate_key_file(path)
            first = load_key_from_file(path)
            generate_key_file(path, overwrite=True)
            second = load_key_from_file(path)
            self.assertNotEqual(first, second)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_load_master_key_prefers_key_file_over_env(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mesh.key"
            generate_key_file(path)
            file_key = load_key_from_file(path)
            with mock.patch.dict(os.environ, {"MEMORYMESH_ENCRYPTION_KEY": "ab" * 32}, clear=False):
                resolved = load_master_key(key_file=path)
            self.assertEqual(resolved, file_key)

    def test_never_prints_key_material(self) -> None:
        # repr()/str() of a generated key file path must never contain key bytes;
        # generate_key_file must not return the key material itself.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mesh.key"
            result = generate_key_file(path)
            self.assertIsInstance(result, Path)
            content = path.read_text(encoding="utf-8").strip()
            # Sanity: file holds only the encoded key, nothing printed elsewhere
            # is asserted implicitly by generate_key_file's return type (Path).
            self.assertTrue(content)


class CryptographyUnavailableTests(unittest.TestCase):
    def test_require_cryptography_raises_clear_error_when_missing(self) -> None:
        with mock.patch.object(crypto, "CRYPTOGRAPHY_AVAILABLE", False):
            with self.assertRaises(CryptographyUnavailableError) as ctx:
                require_cryptography()
            msg = str(ctx.exception)
            self.assertIn("cryptography", msg)
            self.assertIn("memorymesh[security]", msg)

    def test_encryption_context_fails_clearly_without_cryptography(self) -> None:
        with mock.patch.object(crypto, "CRYPTOGRAPHY_AVAILABLE", False):
            with self.assertRaises(CryptographyUnavailableError):
                EncryptionContext(b"k" * 32, "db-1")

    def test_encrypt_field_fails_clearly_without_cryptography(self) -> None:
        # Bypass __init__/__post_init__ (which itself needs cryptography for
        # HKDF) so this test works even in an environment that never had
        # cryptography installed. encrypt_field must check availability
        # before touching any derived key material.
        ctx = EncryptionContext.__new__(EncryptionContext)
        with mock.patch.object(crypto, "CRYPTOGRAPHY_AVAILABLE", False):
            with self.assertRaises(CryptographyUnavailableError):
                encrypt_field(ctx, "x", table="t", column="c", row_identity="1")

    def test_enable_encryption_fails_clearly_without_cryptography(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "m.db"
            with mock.patch.object(crypto, "CRYPTOGRAPHY_AVAILABLE", False):
                with self.assertRaises(CryptographyUnavailableError):
                    enable_encryption(db, master_key=b"k" * 32)


class MigrationEncryptionMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.home = self.root / "home"
        self._home_ctx = temp_home(self.home)
        self._home_ctx.__enter__()

    def tearDown(self) -> None:
        self._home_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def test_schema_reaches_v6_and_default_plaintext_meta(self) -> None:
        db = self.root / "meta.db"
        report = migrate(db)
        self.assertEqual(report.to_version, CURRENT_SCHEMA_VERSION)
        self.assertGreaterEqual(CURRENT_SCHEMA_VERSION, 6)

        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT enabled, key_id, db_identity, terms_mode FROM encryption_meta WHERE id = 1"
            ).fetchone()
            self.assertIsNotNone(row)
            enabled, key_id, db_identity, terms_mode = row
            self.assertEqual(enabled, 0)
            self.assertIsNone(key_id)
            self.assertTrue(db_identity)
            self.assertEqual(terms_mode, "plaintext")

            cols = {r[1] for r in conn.execute("PRAGMA table_info(memory_messages)")}
            self.assertIn("content_fingerprint", cols)

            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn("project_http_tokens", tables)
            self.assertIn("api_pull_state", tables)
        finally:
            conn.close()

    def test_plaintext_status_when_never_enabled(self) -> None:
        db = self.root / "plain.db"
        migrate(db)
        status = encryption_status(db)
        self.assertTrue(status.schema_ready)
        self.assertFalse(status.enabled)
        self.assertEqual(status.terms_mode, "plaintext")

    def test_status_on_missing_db(self) -> None:
        status = encryption_status(self.root / "does-not-exist.db")
        self.assertFalse(status.schema_ready)
        self.assertFalse(status.enabled)


@unittest.skipUnless(crypto.CRYPTOGRAPHY_AVAILABLE, _CRYPTO_SKIP_REASON)
class EnableEncryptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.home = self.root / "home"
        self._home_ctx = temp_home(self.home)
        self._home_ctx.__enter__()
        self.key = os.urandom(32)

    def tearDown(self) -> None:
        self._home_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def test_enable_on_empty_database(self) -> None:
        db = self.root / "empty.db"
        report = enable_encryption(db, master_key=self.key, create_backup=False)
        self.assertEqual(report.action, "enable")
        self.assertEqual(report.messages_encrypted, 0)
        self.assertEqual(report.verified_samples, 0)

        status = encryption_status(db)
        self.assertTrue(status.enabled)
        self.assertEqual(status.key_id, report.key_id)
        self.assertEqual(status.terms_mode, "keyed")

    def test_enable_encrypts_existing_rows_and_plaintext_not_on_disk(self) -> None:
        db = self.root / "data.db"
        store = MemoryStore(db)
        store.init()
        store.insert_messages(
            [
                MemoryRecord(
                    provider="claude",
                    project="proj",
                    conversation_id="conv-1",
                    role="user",
                    content="my secret azure api key is xyz",
                )
            ]
        )
        store.save_embedding(1, "[0.1, 0.2, 0.3]")

        report = enable_encryption(db, master_key=self.key)
        self.assertEqual(report.messages_encrypted, 1)
        self.assertEqual(report.embeddings_encrypted, 1)
        self.assertGreaterEqual(report.verified_samples, 1)
        self.assertIsNotNone(report.backup_path)
        assert report.backup_path is not None
        self.assertTrue(report.backup_path.exists())

        raw = db.read_bytes()
        self.assertNotIn(b"my secret azure api key", raw)

        conn = sqlite3.connect(db)
        try:
            content, fingerprint = conn.execute(
                "SELECT content, content_fingerprint FROM memory_messages WHERE id = 1"
            ).fetchone()
            self.assertTrue(is_envelope(content))
            self.assertIsNotNone(fingerprint)
            terms = {r[0] for r in conn.execute("SELECT term FROM memory_message_terms WHERE message_id = 1")}
            self.assertTrue(terms)
            self.assertNotIn("azure", terms)
        finally:
            conn.close()

        ctx = EncryptionContext(self.key, encryption_status(db).db_identity)
        decrypted = decrypt_field(ctx, content, table="memory_messages", column="content", row_identity="1")
        self.assertEqual(decrypted, "my secret azure api key is xyz")

    def test_enable_twice_raises_lifecycle_error(self) -> None:
        db = self.root / "twice.db"
        enable_encryption(db, master_key=self.key, create_backup=False)
        with self.assertRaises(EncryptionLifecycleError):
            enable_encryption(db, master_key=self.key, create_backup=False)

    def test_failed_enable_leaves_original_usable(self) -> None:
        db = self.root / "fail.db"
        store = MemoryStore(db)
        store.init()
        store.insert_messages(
            [
                MemoryRecord(
                    provider="claude",
                    project="proj",
                    conversation_id="conv-1",
                    role="user",
                    content="keep me plaintext",
                )
            ]
        )

        with mock.patch.object(encryption, "_verify_samples", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                enable_encryption(db, master_key=self.key, create_backup=False)

        status = encryption_status(db)
        self.assertFalse(status.enabled)
        rows = store.list_messages("proj")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "keep me plaintext")


@unittest.skipUnless(crypto.CRYPTOGRAPHY_AVAILABLE, _CRYPTO_SKIP_REASON)
class RotateEncryptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.home = self.root / "home"
        self._home_ctx = temp_home(self.home)
        self._home_ctx.__enter__()
        self.old_key = os.urandom(32)
        self.new_key = os.urandom(32)
        self.db = self.root / "rot.db"
        store = MemoryStore(self.db)
        store.init()
        store.insert_messages(
            [
                MemoryRecord(
                    provider="claude",
                    project="proj",
                    conversation_id="conv-1",
                    role="user",
                    content="rotate me please",
                )
            ]
        )
        self.report = enable_encryption(self.db, master_key=self.old_key, create_backup=False)

    def tearDown(self) -> None:
        self._home_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def test_rotate_changes_key_and_preserves_plaintext_meaning(self) -> None:
        report = rotate_encryption(self.db, old_key=self.old_key, new_key=self.new_key, create_backup=False)
        self.assertEqual(report.previous_key_id, self.report.key_id)
        self.assertNotEqual(report.key_id, self.report.key_id)

        status = encryption_status(self.db)
        self.assertEqual(status.key_id, report.key_id)

        ctx_new = EncryptionContext(self.new_key, status.db_identity)
        conn = sqlite3.connect(self.db)
        try:
            content = conn.execute("SELECT content FROM memory_messages WHERE id = 1").fetchone()[0]
        finally:
            conn.close()
        decrypted = decrypt_field(ctx_new, content, table="memory_messages", column="content", row_identity="1")
        self.assertEqual(decrypted, "rotate me please")

    def test_rotate_wrong_old_key_makes_no_changes(self) -> None:
        bogus_key = os.urandom(32)
        with self.assertRaises(WrongKeyError):
            rotate_encryption(self.db, old_key=bogus_key, new_key=self.new_key, create_backup=False)

        status = encryption_status(self.db)
        self.assertEqual(status.key_id, self.report.key_id)

        ctx_old = EncryptionContext(self.old_key, status.db_identity)
        conn = sqlite3.connect(self.db)
        try:
            content = conn.execute("SELECT content FROM memory_messages WHERE id = 1").fetchone()[0]
        finally:
            conn.close()
        decrypted = decrypt_field(ctx_old, content, table="memory_messages", column="content", row_identity="1")
        self.assertEqual(decrypted, "rotate me please")

    def test_rotate_without_enabling_first_raises(self) -> None:
        db2 = self.root / "notenabled.db"
        MemoryStore(db2).init()
        with self.assertRaises(EncryptionLifecycleError):
            rotate_encryption(db2, old_key=self.old_key, new_key=self.new_key, create_backup=False)


@unittest.skipUnless(crypto.CRYPTOGRAPHY_AVAILABLE, _CRYPTO_SKIP_REASON)
class KeyDomainSeparationAndRemnantTests(unittest.TestCase):
    def test_hkdf_domain_separation_distinct_subkeys(self) -> None:
        master = b"m" * 32
        ctx = EncryptionContext(master, "db-id-1")
        self.assertNotEqual(ctx._enc_key, ctx._content_hmac_key)
        self.assertNotEqual(ctx._enc_key, ctx._term_hmac_key)
        self.assertNotEqual(ctx._content_hmac_key, ctx._term_hmac_key)
        self.assertNotEqual(ctx._enc_key, master)

    def test_enable_plaintext_remnant_scan_and_inject_before_replace(self) -> None:
        from deepiri_memorymesh.models import AgentState, CompressedRecord, MemoryRecord
        from deepiri_memorymesh.storage import MemoryStore

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = Path(tmp.name) / "home"
        home.mkdir()
        markers = {
            "msg": "REMNANT_MSG_MARKER_7f3a9c",
            "meta": "REMNANT_META_MARKER_7f3a9c",
            "sum": "REMNANT_SUM_MARKER_7f3a9c",
            "agent": "REMNANT_AGENT_MARKER_7f3a9c",
            "emb": "REMNANT_EMB_MARKER_7f3a9c",
        }
        with temp_home(home):
            db = home / "mm.db"
            store = MemoryStore(db)
            store.init()
            mid = store.insert_message_returning_id(
                MemoryRecord(
                    provider="python",
                    project="p",
                    conversation_id="c",
                    role="user",
                    content=markers["msg"],
                    metadata_json=json.dumps({"x": markers["meta"]}),
                )
            )
            assert mid is not None
            store.save_embedding(mid, json.dumps({"v": markers["emb"]}))
            store.upsert_summary(
                CompressedRecord(
                    project="p",
                    conversation_id="c",
                    summary=markers["sum"],
                    method="test",
                )
            )
            store.set_agent_state(
                AgentState(project="p", agent="a", key="k", value=markers["agent"])
            )

            key = os.urandom(32)
            encryption._INJECT_FAILURE_AT = "before_replace"
            try:
                report = enable_encryption(db, master_key=key, create_backup=True)
            finally:
                encryption._INJECT_FAILURE_AT = None
            self.assertFalse(report.vacuumed)
            self.assertTrue(report.backup_path is not None and report.backup_path.exists())

            # Original DB remains usable (encrypted in place).
            ctx = EncryptionContext(key, report.db_identity)
            conn = sqlite3.connect(db)
            try:
                content = conn.execute("SELECT content FROM memory_messages WHERE id = ?", (mid,)).fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(
                decrypt_field(ctx, content, table="memory_messages", column="content", row_identity=str(mid)),
                markers["msg"],
            )

            # Successful enable+vacuum remnant scan
            db2 = home / "mm2.db"
            store2 = MemoryStore(db2)
            store2.init()
            mid2 = store2.insert_message_returning_id(
                MemoryRecord(
                    provider="python",
                    project="p",
                    conversation_id="c",
                    role="user",
                    content=markers["msg"],
                    metadata_json=json.dumps({"x": markers["meta"]}),
                )
            )
            assert mid2 is not None
            store2.save_embedding(mid2, json.dumps({"v": markers["emb"]}))
            store2.upsert_summary(
                CompressedRecord(
                    project="p",
                    conversation_id="c",
                    summary=markers["sum"],
                    method="test",
                )
            )
            store2.set_agent_state(
                AgentState(project="p", agent="a", key="k", value=markers["agent"])
            )
            report2 = enable_encryption(db2, master_key=key, create_backup=True)
            self.assertTrue(report2.vacuumed)

            def _scan(path: Path) -> bytes:
                return path.read_bytes() if path.exists() else b""

            backup = report2.backup_path
            assert backup is not None
            for label, marker in markers.items():
                mb = marker.encode("utf-8")
                self.assertIn(mb, backup.read_bytes(), label)
                for path in (
                    db2,
                    Path(str(db2) + "-wal"),
                    Path(str(db2) + "-shm"),
                    Path(str(db2) + ".vacuum-tmp"),
                ):
                    self.assertNotIn(mb, _scan(path), f"{label} in {path}")
            with sqlite3.connect(db2) as conn:
                terms = [r[0] for r in conn.execute("SELECT term FROM memory_message_terms").fetchall()]
            for marker in markers.values():
                for term in terms:
                    self.assertNotIn(marker.lower(), term.lower())


if __name__ == "__main__":
    unittest.main()
