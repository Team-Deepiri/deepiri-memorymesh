"""Batch 4 final review: concurrency, legacy RO, schema detection, reports."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import unittest
import uuid
from pathlib import Path

from deepiri_memorymesh import AmbiguousSchemaError, Memory
from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.embeddings import Embedder
from deepiri_memorymesh.legacy_import import import_legacy_memory
from deepiri_memorymesh.models import CompressedRecord, MemoryRecord
from deepiri_memorymesh.namespace import (
    FACADE_PROVIDER,
    FACADE_ROLE,
    SIMPLE_API_CONVERSATION_ID,
    legacy_import_ownership,
    simple_api_ownership,
)
from deepiri_memorymesh.storage import MemoryStore, detect_db_schema
from deepiri_memorymesh.sync_service import MemoryMesh
from tests.helpers import make_legacy_db, temp_home


class FacadeConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db = Path(self._tmpdir.name) / "mesh.db"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_concurrent_store_dedupes_across_instances(self) -> None:
        barrier = threading.Barrier(8)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=5)
                mem = Memory(db_path=self.db, embedder="fallback")
                mem.store("shared exact content")
            except BaseException as exc:  # noqa: BLE001 — collect for parent
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            self.assertFalse(
                t.is_alive(),
                f"worker thread {t.name} still alive after join(timeout=30)",
            )
        self.assertEqual(errors, [])
        conn = sqlite3.connect(self.db)
        try:
            msg_n = conn.execute("SELECT COUNT(*) FROM memory_messages").fetchone()[0]
            emb_n = conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]
            content_n = conn.execute(
                "SELECT COUNT(*) FROM memory_messages WHERE content = ?",
                ("shared exact content",),
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(msg_n, 1)
        self.assertEqual(emb_n, 1)
        self.assertEqual(content_n, 1)


class FacadeNamespaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db = Path(self._tmpdir.name) / "mesh.db"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_query_searches_only_facade_owned(self) -> None:
        mem = Memory(db_path=self.db, embedder="fallback")
        mem.store("facade azure preference")
        mesh = MemoryMesh(Settings(db_path=self.db, embedding_backend="fallback"))
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="cursor",
                    project="default",
                    conversation_id="other",
                    role="user",
                    content="provider azure preference",
                    timestamp="t",
                )
            ]
        )
        msg_id = int(mesh.store.list_messages("default")[-1]["id"])
        mesh.store.save_embedding(
            msg_id, Embedder("fallback").dumps(Embedder("fallback").embed("provider azure preference"))
        )
        results = mem.query("azure preference", top_k=5)
        self.assertIn("facade azure preference", results)
        self.assertNotIn("provider azure preference", results)
        self.assertEqual(mem.all(), ["facade azure preference"])
        own = simple_api_ownership("default")
        self.assertEqual(own.provider, FACADE_PROVIDER)
        self.assertEqual(own.conversation_id, SIMPLE_API_CONVERSATION_ID)
        self.assertEqual(own.role, FACADE_ROLE)


class SchemaDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_empty_missing_and_zero_byte(self) -> None:
        missing = self.root / "missing.db"
        self.assertEqual(detect_db_schema(missing), "empty")
        zero = self.root / "zero.db"
        zero.write_bytes(b"")
        self.assertEqual(detect_db_schema(zero), "empty")

    def test_canonical_platform(self) -> None:
        db = self.root / "platform.db"
        MemoryStore(db).init()
        self.assertEqual(detect_db_schema(db), "platform")
        before = db.read_bytes()
        detect_db_schema(db)
        self.assertEqual(db.read_bytes(), before)

    def test_exact_legacy_schema(self) -> None:
        db = self.root / "legacy.db"
        make_legacy_db(db, [("a", "[]", "2020-01-01T00:00:00+00:00")])
        self.assertEqual(detect_db_schema(db), "legacy")

    def test_unrelated_memories_table(self) -> None:
        db = self.root / "unrelated.db"
        conn = sqlite3.connect(db)
        try:
            conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, note TEXT)")
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(detect_db_schema(db), "unknown")
        before = db.read_bytes()
        with self.assertRaises(AmbiguousSchemaError):
            Memory(db_path=db, embedder="fallback")
        self.assertEqual(db.read_bytes(), before)

    def test_corrupt_database(self) -> None:
        db = self.root / "corrupt.db"
        db.write_bytes(b"this is not a sqlite database!!!!")
        self.assertEqual(detect_db_schema(db), "corrupt")
        before = db.read_bytes()
        with self.assertRaises(AmbiguousSchemaError):
            Memory(db_path=db, embedder="fallback")
        self.assertEqual(db.read_bytes(), before)

    def test_mixed_legacy_and_canonical(self) -> None:
        db = self.root / "mixed.db"
        MemoryStore(db).init()
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                """
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    embedding TEXT,
                    created_at TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(detect_db_schema(db), "unknown")
        before = db.read_bytes()
        with self.assertRaises(AmbiguousSchemaError):
            MemoryStore(db).init()
        self.assertEqual(db.read_bytes(), before)


class LegacyImportSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.source = self.root / "legacy.db"
        self.dest = self.root / "platform.db"
        make_legacy_db(
            self.source,
            [
                ("iso-row", "[]", "2020-01-01T00:00:00+00:00"),
                ("uuid-row", "[]", str(uuid.uuid4())),
                ("missing-row", "[]", None),
                ("bad-row", "[]", "not-a-timestamp"),
                (None, "[]", "t"),
            ],
        )
        self.before = self.source.read_bytes()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_reject_source_equals_destination(self) -> None:
        with self.assertRaises(ValueError):
            import_legacy_memory(
                source=self.source,
                destination=self.source,
                embedder=Embedder("fallback"),
            )
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_reject_symlink_equivalence(self) -> None:
        link = self.root / "legacy-link.db"
        try:
            os.symlink(self.source, link)
        except (OSError, NotImplementedError) as exc:
            raise unittest.SkipTest(f"symlinks unsupported: {exc}") from exc
        with self.assertRaises(ValueError):
            import_legacy_memory(
                source=link,
                destination=self.source,
                embedder=Embedder("fallback"),
            )
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_dry_run_new_destination_counts(self) -> None:
        report = import_legacy_memory(
            source=self.source,
            destination=self.dest,
            project="p",
            dry_run=True,
            embedder=Embedder("fallback"),
        )
        self.assertEqual(report.imported, 0)
        self.assertEqual(report.importable, 4)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.duplicates_skipped, 0)
        self.assertFalse(self.dest.exists())
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_dry_run_existing_destination_unchanged(self) -> None:
        first = import_legacy_memory(
            source=self.source,
            destination=self.dest,
            project="p",
            dry_run=False,
            embedder=Embedder("fallback"),
        )
        self.assertEqual(first.imported, 4)
        dest_before = self.dest.read_bytes()
        report = import_legacy_memory(
            source=self.source,
            destination=self.dest,
            project="p",
            dry_run=True,
            embedder=Embedder("fallback"),
        )
        self.assertEqual(report.imported, 0)
        self.assertEqual(report.importable, 0)
        self.assertEqual(report.duplicates_skipped, 4)
        self.assertEqual(report.failed, 1)
        self.assertEqual(self.dest.read_bytes(), dest_before)
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_special_character_source_filenames(self) -> None:
        markers = [
            ("space name.db", "space"),
            ("q?mark.db", "qmark"),
            ("hash#tag.db", "hash"),
            ("pct%25.db", "pct"),
            ("unicöde-记忆.db", "unicode"),
        ]
        for filename, token in markers:
            with self.subTest(filename=filename):
                src = self.root / filename
                make_legacy_db(
                    src,
                    [
                        (
                            f"memory from {token}",
                            json.dumps([1.0] + [0.0] * 127),
                            "2020-01-01T00:00:00+00:00",
                        )
                    ],
                )
                before = src.read_bytes()
                dest = self.root / f"dest-{token}.db"
                report = import_legacy_memory(
                    source=src,
                    destination=dest,
                    project="special",
                    embedder=Embedder("fallback"),
                )
                self.assertEqual(report.imported, 1, report.failures)
                self.assertEqual(src.read_bytes(), before)
                mem = Memory(db_path=dest, project="special", embedder="fallback")
                self.assertIn(f"memory from {token}", mem.all())
                self.assertEqual(detect_db_schema(src), "legacy")

    def test_reject_destination_legacy_or_mixed(self) -> None:
        legacy_dest = self.root / "legacy-dest.db"
        make_legacy_db(legacy_dest, [("x", "[]", "t")])
        with self.assertRaises(ValueError):
            import_legacy_memory(
                source=self.source,
                destination=legacy_dest,
                embedder=Embedder("fallback"),
            )

    def test_timestamp_preservation_rules(self) -> None:
        report = import_legacy_memory(
            source=self.source,
            destination=self.dest,
            project="p",
            embedder=Embedder("fallback"),
        )
        self.assertEqual(self.source.read_bytes(), self.before)
        self.assertEqual(report.imported, 4)
        self.assertEqual(report.failed, 1)
        store = MemoryStore(self.dest)
        own = simple_api_ownership("p")
        self.assertEqual(own, legacy_import_ownership("p"))
        rows = {
            r["content"]: r
            for r in store.list_messages_for_namespace(
                project=own.project,
                provider=own.provider,
                conversation_id=own.conversation_id,
                role=own.role,
            )
        }
        self.assertEqual(rows["iso-row"]["timestamp"], "2020-01-01T00:00:00+00:00")
        uuid_ts = str(rows["uuid-row"]["timestamp"])
        self.assertNotRegex(
            uuid_ts,
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        )
        meta_uuid = json.loads(rows["uuid-row"]["metadata_json"])
        self.assertTrue(meta_uuid.get("legacy_created_at_invalid"))
        meta_missing = json.loads(rows["missing-row"]["metadata_json"])
        self.assertTrue(meta_missing.get("legacy_created_at_invalid"))
        meta_bad = json.loads(rows["bad-row"]["metadata_json"])
        self.assertEqual(meta_bad.get("legacy_created_at"), "not-a-timestamp")

    def test_idempotent_across_backends(self) -> None:
        first = import_legacy_memory(
            source=self.source,
            destination=self.dest,
            project="p",
            embedder=Embedder("fallback"),
        )
        self.assertGreater(first.imported, 0)
        before = self.source.read_bytes()

        class FakeEmbedder:
            backend = "sentence-transformers"
            model_id = "fake-model"

            def embed(self, text: str) -> list[float]:
                return [0.5] * 128

            def dumps(self, vector) -> str:
                return Embedder("fallback").dumps(vector)

        second = import_legacy_memory(
            source=self.source,
            destination=self.dest,
            project="p",
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
        )
        self.assertEqual(second.imported, 0)
        self.assertEqual(second.duplicates_skipped, first.imported)
        self.assertEqual(self.source.read_bytes(), before)
        conn = sqlite3.connect(self.dest)
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM memory_messages WHERE project = ?",
                ("p",),
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(n, first.imported)


class BundleReportPrecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.db = self.root / "mesh.db"
        self.bundle = self.root / "bundle.json"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_counts_duplicates_updates_and_nonmutation(self) -> None:
        mesh = MemoryMesh(Settings(db_path=self.db, embedding_backend="fallback"))
        mesh.init()
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="cursor",
                    project="alpha",
                    conversation_id="c1",
                    role="user",
                    content="hello",
                    timestamp="2024-01-01T00:00:00+00:00",
                )
            ]
        )
        mesh.store.upsert_summary(
            CompressedRecord(
                project="alpha",
                conversation_id="c1",
                summary="s1",
                method="m",
                created_at="t1",
            )
        )
        mesh.export_bundle("alpha", self.bundle)
        payload = json.loads(self.bundle.read_text(encoding="utf-8"))
        original_project = payload["project"]
        before = self.bundle.read_bytes()

        dest = MemoryMesh(Settings(db_path=self.root / "dst.db", embedding_backend="fallback"))
        dest.init()
        first = dest.import_bundle_with_report(self.bundle, project_override="beta")
        self.assertEqual(first.messages_inserted, 1)
        self.assertEqual(first.messages_duplicate, 0)
        self.assertEqual(first.summaries_inserted, 1)
        self.assertEqual(first.summaries_updated, 0)
        self.assertEqual(dest.stats("beta")["messages"], 1)
        self.assertEqual(dest.stats("alpha")["messages"], 0)
        # project_override must not mutate parsed/source bundle project
        after_payload = json.loads(self.bundle.read_text(encoding="utf-8"))
        self.assertEqual(after_payload["project"], original_project)
        self.assertEqual(self.bundle.read_bytes(), before)

        second = dest.import_bundle_with_report(self.bundle, project_override="beta")
        self.assertEqual(second.messages_inserted, 0)
        self.assertEqual(second.messages_duplicate, 1)
        self.assertEqual(second.summaries_inserted, 0)
        self.assertEqual(second.summaries_updated, 1)
        self.assertEqual(dest.import_bundle(self.bundle, project_override="beta"), 0)


class LegacyImportFacadeVisibilityTests(unittest.TestCase):
    """Imported legacy rows must use the exact Memory facade ownership tuple."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.source = self.root / "legacy.db"
        self.dest = self.root / "platform.db"
        self.project = "migration-test"
        # Exact historical schema with NOT NULL / UNIQUE where required.
        conn = sqlite3.connect(self.source)
        try:
            conn.execute(
                """
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL UNIQUE,
                    embedding TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO memories (content, embedding, created_at) VALUES (?, ?, ?)",
                (
                    "customer prefers Azure",
                    json.dumps([1.0] + [0.0] * 127),
                    "2020-01-01T00:00:00+00:00",
                ),
            )
            conn.execute(
                "INSERT INTO memories (content, embedding, created_at) VALUES (?, ?, ?)",
                (
                    "working on async refactor",
                    json.dumps([0.0, 1.0] + [0.0] * 126),
                    str(uuid.uuid4()),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        self.source_before = self.source.read_bytes()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_imported_memories_visible_to_memory_facade(self) -> None:
        with temp_home(self.home):
            cfg = Path.home() / ".config" / "deepiri-memorymesh" / "config.yaml"
            self.assertFalse(cfg.exists())

            report = import_legacy_memory(
                source=self.source,
                destination=self.dest,
                project=self.project,
                embedder=Embedder("fallback"),
            )
            self.assertEqual(report.imported, 2)
            self.assertEqual(self.source.read_bytes(), self.source_before)
            self.assertFalse(cfg.exists())

            # Destination is canonical platform schema only.
            self.assertEqual(detect_db_schema(self.dest), "platform")
            conn = sqlite3.connect(self.dest)
            try:
                tables = {
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertIn("memory_messages", tables)
                self.assertNotIn("memories", tables)
                rows = list(
                    conn.execute(
                        """
                        SELECT provider, project, conversation_id, role, content,
                               timestamp, metadata_json
                        FROM memory_messages
                        ORDER BY id ASC
                        """
                    )
                )
                emb_raw = [
                    r[0]
                    for r in conn.execute("SELECT embedding_json FROM memory_embeddings")
                ]
            finally:
                conn.close()

            facade_own = simple_api_ownership(self.project)
            importer_own = legacy_import_ownership(self.project)
            self.assertEqual(facade_own, importer_own)
            self.assertEqual(
                (facade_own.provider, facade_own.project, facade_own.conversation_id, facade_own.role),
                (FACADE_PROVIDER, self.project, SIMPLE_API_CONVERSATION_ID, FACADE_ROLE),
            )
            for row in rows:
                self.assertEqual(row[0], facade_own.provider)
                self.assertEqual(row[1], facade_own.project)
                self.assertEqual(row[2], facade_own.conversation_id)
                self.assertEqual(row[3], facade_own.role)

            by_content = {r[4]: r for r in rows}
            self.assertEqual(
                by_content["customer prefers Azure"][5],
                "2020-01-01T00:00:00+00:00",
            )
            uuid_meta = json.loads(by_content["working on async refactor"][6])
            self.assertTrue(uuid_meta.get("legacy_created_at_invalid"))
            self.assertIn("legacy_created_at", uuid_meta)

            from deepiri_memorymesh.embedding_codec import parse_embedding

            for raw in emb_raw:
                parsed = parse_embedding(raw)
                self.assertFalse(parsed.legacy)

            mem = Memory(db_path=self.dest, project=self.project, embedder="fallback")
            all_contents = mem.all()
            self.assertEqual(
                set(all_contents),
                {"customer prefers Azure", "working on async refactor"},
            )
            self.assertIn("customer prefers Azure", mem.query("Which cloud?", top_k=3))
            self.assertIn("working on async refactor", mem.query("async work", top_k=3))

            mesh = MemoryMesh(
                Settings(db_path=self.dest, embedding_backend="fallback")
            )
            stats = mesh.stats(self.project)
            self.assertEqual(stats["messages"], 2)
            self.assertEqual(stats["embeddings"], 2)
            platform_hits = [
                r["content"]
                for r in mesh.query_with_report(self.project, "Azure cloud").rows
            ]
            self.assertIn("customer prefers Azure", platform_hits)

            mem.store("new facade memory after import")
            self.assertEqual(
                set(mem.all()),
                {
                    "customer prefers Azure",
                    "working on async refactor",
                    "new facade memory after import",
                },
            )
            before_all = list(mem.all())

            again = import_legacy_memory(
                source=self.source,
                destination=self.dest,
                project=self.project,
                embedder=Embedder("fallback"),
            )
            self.assertEqual(again.imported, 0)
            self.assertEqual(again.duplicates_skipped, 2)
            self.assertEqual(self.source.read_bytes(), self.source_before)
            self.assertEqual(mem.all(), before_all)
            conn = sqlite3.connect(self.dest)
            try:
                msg_n = conn.execute(
                    "SELECT COUNT(*) FROM memory_messages WHERE project = ?",
                    (self.project,),
                ).fetchone()[0]
                emb_n = conn.execute(
                    """
                    SELECT COUNT(*) FROM memory_embeddings e
                    JOIN memory_messages m ON e.message_id = m.id
                    WHERE m.project = ?
                    """,
                    (self.project,),
                ).fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(msg_n, 3)  # 2 imported + 1 facade store
            self.assertEqual(emb_n, 3)
            self.assertFalse(cfg.exists())


if __name__ == "__main__":
    unittest.main()
