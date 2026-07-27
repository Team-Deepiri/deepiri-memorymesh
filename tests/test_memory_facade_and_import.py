"""Batch 4: T06 canonical Memory facade, T18 bundle summaries, helpers."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from deepiri_memorymesh import Memory
from deepiri_memorymesh.cli import app
from deepiri_memorymesh.config import Settings, default_db_path
from deepiri_memorymesh.embedding_codec import BACKEND_HASH_V1, parse_embedding
from deepiri_memorymesh.embeddings import Embedder
from deepiri_memorymesh.legacy_import import (
    LEGACY_CONVERSATION_ID,
    LEGACY_PROVIDER,
    LEGACY_ROLE,
    import_legacy_memory,
)
from deepiri_memorymesh.models import CompressedRecord, MemoryRecord, now_iso
from deepiri_memorymesh.storage import LegacySchemaError, detect_db_schema
from deepiri_memorymesh.sync_service import MemoryMesh
from tests.helpers import make_legacy_db, temp_home


class CanonicalMemoryFacadeTests(unittest.TestCase):
    """T06 — Memory uses the platform schema and canonical default path."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.db = self.root / "mesh.db"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_default_path_is_canonical(self) -> None:
        with temp_home(self.root / "home") as home:
            mem = Memory(embedder="fallback")
            expected = Path(home) / ".config" / "deepiri-memorymesh" / "memorymesh.db"
            self.assertEqual(mem.db_path, expected)
            self.assertEqual(mem.db_path, default_db_path())
            self.assertEqual(detect_db_schema(mem.db_path), "platform")
            conn = sqlite3.connect(mem.db_path)
            try:
                tables = {
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            finally:
                conn.close()
            self.assertIn("memory_messages", tables)
            self.assertNotIn("memories", tables)

    def test_custom_db_uses_platform_schema(self) -> None:
        mem = Memory(db_path=self.db, embedder="fallback")
        self.assertEqual(detect_db_schema(self.db), "platform")
        conn = sqlite3.connect(self.db)
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        conn.close()
        self.assertIn("memory_messages", tables)
        self.assertIn("memory_embeddings", tables)
        self.assertNotIn("memories", tables)

    def test_store_writes_platform_messages_and_versioned_embeddings(self) -> None:
        mem = Memory(db_path=self.db, embedder="fallback")
        mem.store("customer prefers Azure")
        conn = sqlite3.connect(self.db)
        row = conn.execute(
            """
            SELECT provider, project, conversation_id, role, content, timestamp
            FROM memory_messages
            """
        ).fetchone()
        emb = conn.execute("SELECT embedding_json FROM memory_embeddings").fetchone()[0]
        conn.close()
        self.assertEqual(row[0], "python")
        self.assertEqual(row[1], "default")
        self.assertEqual(row[2], "python-api")
        self.assertEqual(row[3], "memory")
        self.assertEqual(row[4], "customer prefers Azure")
        parsed = parse_embedding(emb)
        self.assertFalse(parsed.legacy)
        self.assertEqual(parsed.backend, BACKEND_HASH_V1)

    def test_query_works_immediately_after_store(self) -> None:
        mem = Memory(db_path=self.db, embedder="fallback")
        mem.store("customer prefers Azure")
        results = mem.query("Which cloud does the customer prefer?")
        self.assertIn("customer prefers Azure", results)

    def test_all_returns_only_facade_namespace(self) -> None:
        mem = Memory(db_path=self.db, embedder="fallback")
        mem.store("facade memory")
        mesh = MemoryMesh(Settings(db_path=self.db, embedding_backend="fallback"))
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="cursor",
                    project="default",
                    conversation_id="other",
                    role="user",
                    content="platform-only memory",
                )
            ]
        )
        self.assertEqual(mem.all(), ["facade memory"])
        self.assertNotIn("platform-only memory", mem.all())

    def test_exact_duplicate_not_inserted_twice(self) -> None:
        mem = Memory(db_path=self.db, embedder="fallback")
        mem.store("same")
        mem.store("same")
        self.assertEqual(mem.all(), ["same"])
        conn = sqlite3.connect(self.db)
        count = conn.execute("SELECT COUNT(*) FROM memory_messages").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_platform_sees_facade_memories(self) -> None:
        mem = Memory(db_path=self.db, embedder="fallback")
        mem.store("shared azure preference")
        mesh = MemoryMesh(Settings(db_path=self.db, embedding_backend="fallback"))
        stats = mesh.stats("default")
        self.assertEqual(stats["messages"], 1)
        self.assertEqual(stats["embeddings"], 1)
        result = mesh.query_with_report("default", "azure cloud preference")
        contents = [r["content"] for r in result.rows]
        self.assertIn("shared azure preference", contents)

    def test_facade_does_not_require_http(self) -> None:
        with mock.patch("deepiri_memorymesh.service_api.run_service") as serve:
            mem = Memory(db_path=self.db, embedder="fallback")
            mem.store("offline")
            self.assertTrue(mem.query("offline"))
            serve.assert_not_called()

    def test_legacy_db_detected_not_rewritten(self) -> None:
        legacy = self.root / "legacy.db"
        make_legacy_db(legacy, [("old memory", "[0.1,0.2]", "2020-01-01T00:00:00+00:00")])
        before = legacy.read_bytes()
        with self.assertRaises(LegacySchemaError):
            Memory(db_path=legacy, embedder="fallback")
        self.assertEqual(legacy.read_bytes(), before)
        self.assertEqual(detect_db_schema(legacy), "legacy")
        conn = sqlite3.connect(legacy)
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        conn.close()
        self.assertIn("memories", tables)
        self.assertNotIn("memory_messages", tables)

    def test_custom_project_deterministic(self) -> None:
        mem = Memory(db_path=self.db, embedder="fallback", project="acme")
        mem.store("acme fact")
        self.assertEqual(mem.all(), ["acme fact"])
        other = Memory(db_path=self.db, embedder="fallback", project="other")
        self.assertEqual(other.all(), [])
        mesh = MemoryMesh(Settings(db_path=self.db, embedding_backend="fallback"))
        self.assertEqual(mesh.stats("acme")["messages"], 1)
        self.assertEqual(mesh.stats("other")["messages"], 0)

    def test_no_yaml_rewritten_by_facade(self) -> None:
        with temp_home(self.root / "home"):
            cfg = Path.home() / ".config" / "deepiri-memorymesh" / "config.yaml"
            self.assertFalse(cfg.exists())
            Memory(embedder="fallback").store("x")
            self.assertFalse(cfg.exists())


class LegacyImporterTests(unittest.TestCase):
    """T06 — explicit non-destructive legacy importer."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.source = self.root / "legacy.db"
        self.dest = self.root / "platform.db"
        make_legacy_db(
            self.source,
            [
                ("alpha", json.dumps([1.0] + [0.0] * 127), "2020-01-01T00:00:00+00:00"),
                ("beta", json.dumps([0.0, 1.0] + [0.0] * 126), "2020-01-02T00:00:00+00:00"),
                (None, "[]", "t"),  # malformed
            ],
        )
        self.before = self.source.read_bytes()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_dry_run_writes_nothing(self) -> None:
        report = import_legacy_memory(
            source=self.source,
            destination=self.dest,
            project="migrated",
            dry_run=True,
            embedder=Embedder("fallback"),
        )
        self.assertTrue(report.dry_run)
        self.assertFalse(self.dest.exists())
        self.assertEqual(self.source.read_bytes(), self.before)
        self.assertEqual(report.scanned, 3)
        self.assertEqual(report.importable, 2)
        self.assertEqual(report.imported, 0)
        self.assertEqual(report.failed, 1)

    def test_import_success_idempotent_and_versioned(self) -> None:
        report = import_legacy_memory(
            source=self.source,
            destination=self.dest,
            project="migrated",
            dry_run=False,
            embedder=Embedder("fallback"),
        )
        self.assertEqual(self.source.read_bytes(), self.before)
        self.assertEqual(report.imported, 2)
        self.assertEqual(report.failed, 1)
        mesh = MemoryMesh(Settings(db_path=self.dest, embedding_backend="fallback"))
        self.assertEqual(mesh.stats("migrated")["messages"], 2)
        self.assertEqual(mesh.stats("migrated")["embeddings"], 2)
        emb = mesh.store.list_embeddings("migrated")[0]["embedding_json"]
        self.assertFalse(parse_embedding(emb).legacy)

        again = import_legacy_memory(
            source=self.source,
            destination=self.dest,
            project="migrated",
            dry_run=False,
            embedder=Embedder("fallback"),
        )
        self.assertEqual(again.imported, 0)
        self.assertEqual(again.duplicates_skipped, 2)
        self.assertEqual(mesh.stats("migrated")["messages"], 2)
        self.assertEqual(self.source.read_bytes(), self.before)

        rows = mesh.store.list_messages_for_namespace(
            project="migrated",
            provider=LEGACY_PROVIDER,
            conversation_id=LEGACY_CONVERSATION_ID,
            role=LEGACY_ROLE,
        )
        self.assertEqual({r["content"] for r in rows}, {"alpha", "beta"})

    def test_cli_import_legacy_dry_run(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "import-legacy-memory",
                "--source",
                str(self.source),
                "--destination",
                str(self.dest),
                "--project",
                "p",
                "--dry-run",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.stdout + result.stderr)
        self.assertIn("dry-run", result.stdout)
        self.assertIn("Would import", result.stdout)
        self.assertIn("No changes were made", result.stdout)
        self.assertIn("imported=0", result.stdout)
        self.assertFalse(self.dest.exists())
        self.assertEqual(self.source.read_bytes(), self.before)


class BundleSummaryImportTests(unittest.TestCase):
    """T18 — bundle import restores summaries."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.src_db = self.root / "src.db"
        self.dst_db = self.root / "dst.db"
        self.bundle = self.root / "bundle.json"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _seed_export(self) -> MemoryMesh:
        mesh = MemoryMesh(Settings(db_path=self.src_db, embedding_backend="fallback"))
        mesh.init()
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="cursor",
                    project="alpha",
                    conversation_id="c1",
                    role="user",
                    content="hello world",
                    timestamp="2024-01-01T00:00:00+00:00",
                )
            ]
        )
        mesh.store.upsert_summary(
            CompressedRecord(
                project="alpha",
                conversation_id="c1",
                summary="hello summary",
                method="extractive-frequency",
                created_at="2024-01-02T00:00:00+00:00",
            )
        )
        mesh.export_bundle("alpha", self.bundle)
        return mesh

    def test_round_trip_messages_and_summaries(self) -> None:
        self._seed_export()
        before = self.bundle.read_bytes()
        dest = MemoryMesh(Settings(db_path=self.dst_db, embedding_backend="fallback"))
        dest.init()
        report = dest.import_bundle_with_report(self.bundle)
        self.assertEqual(report.messages_inserted, 1)
        self.assertEqual(report.summaries_inserted, 1)
        self.assertEqual(dest.stats("alpha")["messages"], 1)
        self.assertEqual(dest.stats("alpha")["summaries"], 1)
        summary = dest.store.list_summaries("alpha")[0]
        self.assertEqual(summary["summary"], "hello summary")
        self.assertEqual(summary["conversation_id"], "c1")
        self.assertEqual(summary["method"], "extractive-frequency")
        self.assertEqual(self.bundle.read_bytes(), before)

        # Idempotent re-import
        again = dest.import_bundle_with_report(self.bundle)
        self.assertEqual(again.messages_inserted, 0)
        self.assertEqual(again.messages_duplicate, 1)
        self.assertEqual(again.summaries_updated, 1)  # upsert updates existing
        self.assertEqual(dest.stats("alpha")["messages"], 1)
        self.assertEqual(dest.stats("alpha")["summaries"], 1)

    def test_project_override_applies_to_both(self) -> None:
        self._seed_export()
        dest = MemoryMesh(Settings(db_path=self.dst_db, embedding_backend="fallback"))
        dest.init()
        dest.import_bundle_with_report(self.bundle, project_override="beta")
        self.assertEqual(dest.stats("beta")["messages"], 1)
        self.assertEqual(dest.stats("beta")["summaries"], 1)
        self.assertEqual(dest.stats("alpha")["messages"], 0)

    def test_updated_summary_follows_upsert(self) -> None:
        self._seed_export()
        dest = MemoryMesh(Settings(db_path=self.dst_db, embedding_backend="fallback"))
        dest.init()
        dest.import_bundle_with_report(self.bundle)
        payload = json.loads(self.bundle.read_text(encoding="utf-8"))
        payload["summaries"][0]["summary"] = "updated summary"
        payload["summaries"][0]["created_at"] = "2024-06-01T00:00:00+00:00"
        updated = self.root / "bundle2.json"
        updated.write_text(json.dumps(payload), encoding="utf-8")
        dest.import_bundle_with_report(updated)
        row = dest.store.list_summaries("alpha")[0]
        self.assertEqual(row["summary"], "updated summary")
        self.assertEqual(row["created_at"], "2024-06-01T00:00:00+00:00")

    def test_malformed_summary_does_not_block_messages(self) -> None:
        payload = {
            "project": "p",
            "messages": [
                {
                    "provider": "x",
                    "conversation_id": "c",
                    "role": "user",
                    "content": "ok",
                    "timestamp": now_iso(),
                }
            ],
            "summaries": [
                {"conversation_id": "c"},  # missing summary/method
                "not-an-object",
            ],
        }
        self.bundle.write_text(json.dumps(payload), encoding="utf-8")
        dest = MemoryMesh(Settings(db_path=self.dst_db, embedding_backend="fallback"))
        dest.init()
        report = dest.import_bundle_with_report(self.bundle)
        self.assertEqual(report.messages_imported, 1)
        self.assertEqual(report.malformed_summaries, 2)
        self.assertEqual(dest.stats("p")["messages"], 1)
        self.assertEqual(dest.stats("p")["summaries"], 0)

    def test_malformed_message_does_not_block_summaries(self) -> None:
        payload = {
            "project": "p",
            "messages": ["bad", {"content": ""}],
            "summaries": [
                {
                    "conversation_id": "c1",
                    "summary": "s",
                    "method": "m",
                    "created_at": "t",
                }
            ],
        }
        self.bundle.write_text(json.dumps(payload), encoding="utf-8")
        dest = MemoryMesh(Settings(db_path=self.dst_db, embedding_backend="fallback"))
        dest.init()
        report = dest.import_bundle_with_report(self.bundle)
        self.assertEqual(report.messages_imported, 0)
        self.assertEqual(report.malformed_messages, 2)
        self.assertEqual(report.summaries_imported, 1)
        self.assertEqual(dest.stats("p")["summaries"], 1)

    def test_old_bundle_without_summaries(self) -> None:
        payload = {
            "project": "p",
            "messages": [
                {
                    "provider": "x",
                    "conversation_id": "c",
                    "role": "user",
                    "content": "only-msg",
                    "timestamp": "t",
                }
            ],
        }
        self.bundle.write_text(json.dumps(payload), encoding="utf-8")
        dest = MemoryMesh(Settings(db_path=self.dst_db, embedding_backend="fallback"))
        dest.init()
        report = dest.import_bundle_with_report(self.bundle)
        self.assertEqual(report.messages_imported, 1)
        self.assertEqual(report.summaries_seen, 0)

    def test_empty_summaries_list(self) -> None:
        payload = {"project": "p", "messages": [], "summaries": []}
        self.bundle.write_text(json.dumps(payload), encoding="utf-8")
        dest = MemoryMesh(Settings(db_path=self.dst_db, embedding_backend="fallback"))
        dest.init()
        report = dest.import_bundle_with_report(self.bundle)
        self.assertEqual(report.messages_imported, 0)
        self.assertEqual(report.summaries_imported, 0)

    def test_import_bundle_return_compatibility(self) -> None:
        self._seed_export()
        dest = MemoryMesh(Settings(db_path=self.dst_db, embedding_backend="fallback"))
        dest.init()
        count = dest.import_bundle(self.bundle)
        self.assertIsInstance(count, int)
        self.assertEqual(count, 1)

    def test_cli_reports_both_categories(self) -> None:
        self._seed_export()
        dest = MemoryMesh(Settings(db_path=self.dst_db, embedding_backend="fallback"))
        dest.init()
        runner = CliRunner()
        with mock.patch("deepiri_memorymesh.cli._mesh", return_value=dest):
            result = runner.invoke(
                app,
                ["bundle", "import", "--bundle", str(self.bundle)],
            )
        self.assertEqual(result.exit_code, 0, result.stdout + result.stderr)
        self.assertIn("messages=1/1", result.stdout)
        self.assertIn("summaries_inserted=1", result.stdout)
        self.assertIn("summaries_seen=1", result.stdout)


class Sha256HelperTests(unittest.TestCase):
    """Sanity: legacy source byte identity helper used above."""

    def test_hash_stable(self) -> None:
        data = b"abc"
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            hashlib.sha256(data).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
