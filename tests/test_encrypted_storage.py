"""Batch 6: T33 runtime storage encryption (reads/writes/search/facade/bundles)."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import temp_home

from deepiri_memorymesh import crypto
from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.crypto import generate_key_file, load_key_from_file
from deepiri_memorymesh.encryption import enable_encryption
from deepiri_memorymesh.models import MemoryRecord
from deepiri_memorymesh.storage import MemoryStore
from deepiri_memorymesh.sync_service import MemoryMesh
from memorymesh import Memory

_CRYPTO_SKIP = "cryptography not installed; pip install '.[security]'"


@unittest.skipUnless(crypto.CRYPTOGRAPHY_AVAILABLE, _CRYPTO_SKIP)
class RuntimeStorageEncryptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self.db = self.home / "mm.db"
        self.key_path = self.home / "enc.key"
        generate_key_file(self.key_path)
        self.key = load_key_from_file(self.key_path)
        self._home_cm = temp_home(self.home)
        self._home_cm.__enter__()
        self._env = mock.patch.dict(
            os.environ,
            {
                "MEMORYMESH_ENCRYPTION_KEY": self.key_path.read_text(encoding="utf-8").strip(),
            },
            clear=False,
        )
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        self._home_cm.__exit__(None, None, None)
        self.tmp.cleanup()

    def _enable(self, store: MemoryStore | None = None) -> MemoryStore:
        store = store or MemoryStore(self.db, key_file=self.key_path)
        store.init()
        # Seed plaintext then encrypt in place.
        store.insert_messages(
            [
                MemoryRecord(
                    provider="python",
                    project="p1",
                    conversation_id="c1",
                    role="memory",
                    content="secret transcript ALPHA unique",
                    metadata_json='{"note":"meta-secret"}',
                )
            ]
        )
        enable_encryption(self.db, master_key=self.key)
        store._invalidate_crypto_cache()
        return store

    def test_sensitive_fields_encrypted_at_raw_sqlite(self) -> None:
        store = self._enable()
        with store.connection() as conn:
            row = conn.execute("SELECT content, metadata_json FROM memory_messages").fetchone()
            self.assertIn('"alg":"AES-256-GCM"', row["content"])
            self.assertNotIn("secret transcript", row["content"])
            self.assertNotIn("meta-secret", row["metadata_json"])
            terms = [
                r[0]
                for r in conn.execute("SELECT term FROM memory_message_terms").fetchall()
            ]
            joined = " ".join(terms)
            self.assertNotIn("secret", joined)
            self.assertNotIn("transcript", joined)
            self.assertNotIn("alpha", joined.lower())

    def test_list_returns_plaintext_and_facade_dedupe(self) -> None:
        store = self._enable()
        rows = store.list_messages("p1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["content"], "secret transcript ALPHA unique")

        mem = Memory(db_path=self.db, project="facade", embedder="fallback")
        # Enable on this DB already done — facade uses same key via env.
        # Store into encrypted DB: need encryption enabled already.
        enable_encryption  # noqa: B018 - already enabled above
        mem.store("hello facade one")
        mem.store("hello facade one")
        self.assertEqual(mem.all().count("hello facade one"), 1)
        results = mem.query("hello facade", top_k=3, strategy="exact")
        self.assertTrue(any("hello facade" in r for r in results))

    def test_indexed_retrieval_with_keyed_terms(self) -> None:
        store = MemoryStore(self.db, key_file=self.key_path)
        store.init()
        settings = Settings(db_path=self.db, embedding_backend="fallback", encryption_key_file=self.key_path)
        mesh = MemoryMesh(settings)
        mesh.init()
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="jsonl",
                    project="proj",
                    conversation_id="conv",
                    role="user",
                    content="zebra pineapple retrieval marker",
                )
            ]
        )
        mid = mesh.store.list_messages("proj")[0]["id"]
        mesh.store.save_embedding(mid, mesh.embedder.dumps(mesh.embedder.embed("zebra pineapple retrieval marker")))
        enable_encryption(self.db, master_key=self.key)
        mesh.store._invalidate_crypto_cache()

        result = mesh.query_with_report("proj", "pineapple retrieval", top_k=5, strategy="indexed")
        self.assertGreaterEqual(len(result.rows), 1)
        self.assertIn("pineapple", result.rows[0]["content"])

    def test_encrypted_bundle_roundtrip_and_plaintext_opt_in(self) -> None:
        store = MemoryStore(self.db, key_file=self.key_path)
        store.init()
        settings = Settings(db_path=self.db, embedding_backend="fallback", encryption_key_file=self.key_path)
        mesh = MemoryMesh(settings)
        mesh.init()
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="bundle",
                    project="bp",
                    conversation_id="bc",
                    role="user",
                    content="bundle secret content ZZZ",
                )
            ]
        )
        enable_encryption(self.db, master_key=self.key)
        mesh.store._invalidate_crypto_cache()

        out = Path(self.tmp.name) / "b.json"
        mesh.export_bundle("bp", out)
        raw = out.read_text(encoding="utf-8")
        self.assertIn('"encrypted": true', raw)
        self.assertNotIn("bundle secret content ZZZ", raw)

        # Import into a fresh encrypted DB using the same key.
        db2 = Path(self.tmp.name) / "mm2.db"
        store2 = MemoryStore(db2, key_file=self.key_path)
        store2.init()
        # Copy encryption by enabling empty DB with same key, or insert then enable.
        # Empty enable works:
        enable_encryption(db2, master_key=self.key)
        settings2 = Settings(db_path=db2, embedding_backend="fallback", encryption_key_file=self.key_path)
        mesh2 = MemoryMesh(settings2)
        mesh2.init()
        report = mesh2.import_bundle_with_report(out)
        self.assertEqual(report.messages_inserted, 1)
        self.assertEqual(mesh2.store.list_messages("bp")[0]["content"], "bundle secret content ZZZ")

        plain = Path(self.tmp.name) / "plain.json"
        with self.assertWarns(UserWarning):
            mesh.export_bundle("bp", plain, allow_plaintext=True)
        self.assertIn("bundle secret content ZZZ", plain.read_text(encoding="utf-8"))

    def test_project_isolation_fingerprint(self) -> None:
        store = MemoryStore(self.db, key_file=self.key_path)
        store.init()
        enable_encryption(self.db, master_key=self.key)
        store._invalidate_crypto_cache()
        shared = "same plaintext across projects"
        store.insert_messages(
            [
                MemoryRecord(
                    provider="python",
                    project="a",
                    conversation_id="python-api",
                    role="memory",
                    content=shared,
                ),
                MemoryRecord(
                    provider="python",
                    project="b",
                    conversation_id="python-api",
                    role="memory",
                    content=shared,
                ),
            ]
        )
        self.assertEqual(len(store.list_messages("a")), 1)
        self.assertEqual(len(store.list_messages("b")), 1)
        with store.connection() as conn:
            fps = [
                r[0]
                for r in conn.execute(
                    "SELECT content_fingerprint FROM memory_messages ORDER BY project"
                ).fetchall()
            ]
        self.assertEqual(len(fps), 2)
        self.assertNotEqual(fps[0], fps[1])

    def test_encrypt_before_insert_no_plaintext_in_db_bytes(self) -> None:
        store = MemoryStore(self.db, key_file=self.key_path)
        store.init()
        enable_encryption(self.db, master_key=self.key)
        store._invalidate_crypto_cache()
        marker = "PLAINTEXT_MARKER_NEVER_ON_DISK_XYZ_9911"
        store.insert_messages(
            [
                MemoryRecord(
                    provider="python",
                    project="p1",
                    conversation_id="c1",
                    role="memory",
                    content=marker,
                    metadata_json=f'{{"m":"{marker}"}}',
                )
            ]
        )
        raw = self.db.read_bytes()
        self.assertNotIn(marker.encode("utf-8"), raw)
        with store.connection() as conn:
            row = conn.execute("SELECT content, metadata_json FROM memory_messages").fetchone()
            self.assertNotIn(marker, row["content"])
            self.assertNotIn(marker, row["metadata_json"])
            self.assertIn('"alg":"AES-256-GCM"', row["content"])


if __name__ == "__main__":
    unittest.main()
