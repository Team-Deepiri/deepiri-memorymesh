"""Batch 5: T16 migrations, T13 WAL/concurrency, T20 Aider, T19 registry, T32 index, T34 TUI."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from tests.helpers import make_legacy_db, temp_home

from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.migrations import (
    CURRENT_SCHEMA_VERSION,
    MigrationError,
    migrate,
    migration_status,
)
from deepiri_memorymesh.models import MemoryRecord, now_iso
from deepiri_memorymesh.providers.aider import (
    conversation_id_for_path,
    parse_aider_file,
    parse_aider_input_history,
)
from deepiri_memorymesh.providers.registry import get_provider, list_providers, native_parser_map
from deepiri_memorymesh.search_index import (
    search_index_status,
    tokenize_for_index,
)
from deepiri_memorymesh.storage import MemoryStore, detect_db_schema
from deepiri_memorymesh.supervised_service import SupervisedService, detect_existing_service
from deepiri_memorymesh.sync_service import MemoryMesh
from memorymesh import Memory


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "aider"


def _batch4_schema_script() -> str:
    return """
    CREATE TABLE memory_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT NOT NULL,
        project TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        metadata_json TEXT NOT NULL
    );
    CREATE UNIQUE INDEX ux_messages_dedupe
    ON memory_messages (project, provider, conversation_id, role, content, timestamp);
    CREATE TABLE memory_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        summary TEXT NOT NULL,
        method TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE memory_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER NOT NULL,
        embedding_json TEXT NOT NULL,
        FOREIGN KEY(message_id) REFERENCES memory_messages(id)
    );
    CREATE UNIQUE INDEX ux_embeddings_message ON memory_embeddings (message_id);
    CREATE TABLE agent_state (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project TEXT NOT NULL,
        agent TEXT NOT NULL,
        state_key TEXT NOT NULL,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(project, agent, state_key)
    );
    """


class MigrationFrameworkTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.home = self.root / "home"
        self._home_ctx = temp_home(self.home)
        self._home_ctx.__enter__()

    def tearDown(self) -> None:
        self._home_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def test_new_database_reaches_latest(self) -> None:
        db = self.root / "new.db"
        report = migrate(db)
        self.assertEqual(report.to_version, CURRENT_SCHEMA_VERSION)
        self.assertFalse(report.no_change)
        store = MemoryStore(db)
        status = store.database_status()
        self.assertEqual(status.schema_version, CURRENT_SCHEMA_VERSION)
        self.assertTrue(status.foreign_keys)

    def test_unversioned_batch4_adopted_without_row_loss(self) -> None:
        db = self.root / "batch4.db"
        conn = sqlite3.connect(db)
        try:
            conn.executescript(_batch4_schema_script())
            conn.execute(
                """
                INSERT INTO memory_messages
                (provider, project, conversation_id, role, content, timestamp, metadata_json)
                VALUES ('claude','p','c','user','hello','2024-01-01T00:00:00+00:00','{}')
                """
            )
            mid = conn.execute("SELECT id FROM memory_messages").fetchone()[0]
            conn.execute(
                "INSERT INTO memory_embeddings (message_id, embedding_json) VALUES (?, ?)",
                (mid, "[0.1,0.2]"),
            )
            conn.execute(
                """
                INSERT INTO memory_summaries
                (project, conversation_id, summary, method, created_at)
                VALUES ('p','c','sum','m','2024-01-01T00:00:00+00:00')
                """
            )
            conn.execute(
                """
                INSERT INTO agent_state (project, agent, state_key, value, updated_at)
                VALUES ('p','a','k','v','2024-01-01T00:00:00+00:00')
                """
            )
            conn.commit()
        finally:
            conn.close()
        before = db.read_bytes()
        report = migrate(db)
        self.assertTrue(report.adopted_unversioned_baseline or report.applied)
        self.assertEqual(report.to_version, CURRENT_SCHEMA_VERSION)
        conn = sqlite3.connect(db)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM memory_messages").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM memory_summaries").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM agent_state").fetchone()[0], 1)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(memory_messages)")}
            self.assertIn("source_key", cols)
            self.assertIn("memory_message_terms", {r[0] for r in conn.execute("SELECT name FROM sqlite_master")})
        finally:
            conn.close()
        # Original content still present (schema evolved; bytes may differ).
        self.assertNotEqual(before, db.read_bytes())

    def test_legacy_rejected(self) -> None:
        db = self.root / "legacy.db"
        make_legacy_db(db, [("x", "[]", "t")])
        with self.assertRaises(MigrationError):
            migrate(db)

    def test_unrelated_rejected(self) -> None:
        db = self.root / "other.db"
        conn = sqlite3.connect(db)
        try:
            conn.execute("CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT)")
            conn.commit()
        finally:
            conn.close()
        with self.assertRaises(MigrationError):
            migrate(db)

    def test_mixed_rejected(self) -> None:
        db = self.root / "mixed.db"
        conn = sqlite3.connect(db)
        try:
            conn.executescript(_batch4_schema_script())
            conn.execute(
                "CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT, embedding TEXT, created_at TEXT)"
            )
            conn.commit()
        finally:
            conn.close()
        self.assertEqual(detect_db_schema(db), "unknown")
        with self.assertRaises(MigrationError):
            migrate(db)

    def test_corrupt_rejected(self) -> None:
        db = self.root / "corrupt.db"
        db.write_bytes(b"not a sqlite database!!!!")
        with self.assertRaises((MigrationError, sqlite3.Error)):
            migrate(db)

    def test_future_version_rejected(self) -> None:
        db = self.root / "future.db"
        migrate(db)
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (999,'future',?)",
                (now_iso(),),
            )
            conn.execute("PRAGMA user_version = 999")
            conn.commit()
        finally:
            conn.close()
        with self.assertRaises(MigrationError):
            migrate(db)

    def test_missing_history_version_rejected(self) -> None:
        db = self.root / "gap.db"
        migrate(db)
        conn = sqlite3.connect(db)
        try:
            conn.execute("DELETE FROM schema_migrations WHERE version = 2")
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
        finally:
            conn.close()
        with self.assertRaises(MigrationError):
            migrate(db)

    def test_migration_failure_rolls_back_and_not_recorded(self) -> None:
        db = self.root / "fail.db"
        migrate(db)  # up to date
        last_version = CURRENT_SCHEMA_VERSION
        # Force a fake pending migration by deleting the latest version's
        # history then patching its apply function to fail. Generalized over
        # CURRENT_SCHEMA_VERSION so this keeps working as migrations are added.
        conn = sqlite3.connect(db)
        try:
            conn.execute("DELETE FROM schema_migrations WHERE version = ?", (last_version,))
            conn.execute("DROP TABLE api_pull_state")
            conn.execute(f"PRAGMA user_version = {last_version - 1}")
            conn.commit()
        finally:
            conn.close()

        def boom(_conn: sqlite3.Connection) -> None:
            raise RuntimeError("boom")

        from deepiri_memorymesh import migrations as migmod

        original = list(migmod.MIGRATIONS)
        original_last = original[-1]
        try:
            patched = [m for m in original if m.version < last_version]
            from deepiri_memorymesh.migrations import Migration

            patched.append(Migration(last_version, original_last.name, boom))
            with mock.patch.object(migmod, "MIGRATIONS", patched):
                with mock.patch.dict(
                    migmod._MIGRATIONS_BY_VERSION, {last_version: patched[-1]}, clear=False
                ):
                    with self.assertRaises(MigrationError) as ctx:
                        migrate(db)
            self.assertEqual(ctx.exception.version, last_version)
        finally:
            pass
        conn = sqlite3.connect(db)
        try:
            versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations")]
            self.assertNotIn(last_version, versions)
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn("api_pull_state", tables)
        finally:
            conn.close()

    def test_repeated_migrate_noop(self) -> None:
        db = self.root / "noop.db"
        first = migrate(db)
        second = migrate(db)
        self.assertFalse(first.no_change)
        self.assertTrue(second.no_change)
        self.assertEqual(second.applied, [])

    def test_concurrent_migrate_once(self) -> None:
        db = self.root / "conc.db"
        errors: list[BaseException] = []
        reports: list = []

        def worker() -> None:
            try:
                reports.append(migrate(db))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])
        applied_runs = [r for r in reports if r.applied]
        self.assertGreaterEqual(len(applied_runs), 1)
        conn = sqlite3.connect(db)
        try:
            rows = conn.execute("SELECT version, name FROM schema_migrations ORDER BY version").fetchall()
            self.assertEqual([r[0] for r in rows], list(range(1, CURRENT_SCHEMA_VERSION + 1)))
        finally:
            conn.close()

    def test_concurrent_unversioned_upgrade_single_backup(self) -> None:
        db = self.root / "batch4_conc.db"
        conn = sqlite3.connect(db)
        try:
            conn.executescript(_batch4_schema_script())
            conn.execute(
                """
                INSERT INTO memory_messages
                (provider, project, conversation_id, role, content, timestamp, metadata_json)
                VALUES ('claude','p','c','user','keep','2024-01-01T00:00:00+00:00','{}')
                """
            )
            conn.commit()
        finally:
            conn.close()

        reports: list = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def worker() -> None:
            try:
                barrier.wait(timeout=10)
                reports.append(migrate(db))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])
        backups = list(db.parent.glob(f"{db.name}.pre-migrate-v0.*.bak"))
        self.assertEqual(len(backups), 1, backups)
        applied = [r for r in reports if r.applied]
        noops = [r for r in reports if r.no_change]
        self.assertEqual(len(applied), 1)
        self.assertEqual(len(noops), 1)
        self.assertIsNotNone(applied[0].backup_path)
        self.assertIsNone(noops[0].backup_path)
        bconn = sqlite3.connect(backups[0])
        try:
            tables = {r[0] for r in bconn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn("memory_messages", tables)
            self.assertNotIn("schema_migrations", tables)
            self.assertEqual(
                bconn.execute("SELECT content FROM memory_messages").fetchone()[0],
                "keep",
            )
        finally:
            bconn.close()
        conn = sqlite3.connect(db)
        try:
            versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
            self.assertEqual(versions, list(range(1, CURRENT_SCHEMA_VERSION + 1)))
            # Each migration recorded exactly once.
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0],
                CURRENT_SCHEMA_VERSION,
            )
        finally:
            conn.close()

    def test_backup_created_for_real_upgrade(self) -> None:
        db = self.root / "backup.db"
        conn = sqlite3.connect(db)
        try:
            conn.executescript(_batch4_schema_script())
            conn.execute(
                """
                INSERT INTO memory_messages
                (provider, project, conversation_id, role, content, timestamp, metadata_json)
                VALUES ('claude','p','c','user','keep-me','2024-01-01T00:00:00+00:00','{}')
                """
            )
            conn.commit()
        finally:
            conn.close()
        report = migrate(db)
        self.assertIsNotNone(report.backup_path)
        assert report.backup_path is not None
        self.assertTrue(report.backup_path.exists())
        bconn = sqlite3.connect(report.backup_path)
        try:
            content = bconn.execute("SELECT content FROM memory_messages").fetchone()[0]
            self.assertEqual(content, "keep-me")
            tables = {r[0] for r in bconn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn("schema_migrations", tables)
        finally:
            bconn.close()

    def test_dry_run_and_status_no_writes(self) -> None:
        db = self.root / "dry.db"
        conn = sqlite3.connect(db)
        try:
            conn.executescript(_batch4_schema_script())
            conn.commit()
        finally:
            conn.close()
        before = db.read_bytes()
        status = migration_status(db)
        self.assertTrue(status.pending)
        dry = migrate(db, dry_run=True)
        self.assertTrue(dry.dry_run)
        self.assertEqual(db.read_bytes(), before)

    def test_readonly_detection_non_mutating(self) -> None:
        db = self.root / "ro.db"
        migrate(db)
        before = db.read_bytes()
        self.assertEqual(detect_db_schema(db), "platform")
        migration_status(db)
        self.assertEqual(db.read_bytes(), before)


class ConcurrencyWalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.db = self.root / "c.db"
        self.home = self.root / "home"
        self._home_ctx = temp_home(self.home)
        self._home_ctx.__enter__()
        MemoryStore(self.db).init()

    def tearDown(self) -> None:
        self._home_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def test_wal_and_busy_timeout(self) -> None:
        status = MemoryStore(self.db).database_status()
        self.assertEqual(status.journal_mode, "wal")
        self.assertTrue(status.wal_active)
        self.assertTrue(status.foreign_keys)
        self.assertGreaterEqual(status.busy_timeout_ms, 1000)

    def test_concurrent_facade_stores(self) -> None:
        barrier = threading.Barrier(8)
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                barrier.wait(timeout=10)
                mem = Memory(db_path=self.db, embedder="fallback")
                mem.store(f"unique-content-{i}")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])
        mem = Memory(db_path=self.db, embedder="fallback")
        self.assertEqual(len(mem.all()), 8)

    def test_concurrent_reads_during_writes(self) -> None:
        mesh = MemoryMesh(Settings(db_path=self.db, embedding_backend="fallback"))
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="python",
                    project="default",
                    conversation_id="python-api",
                    role="memory",
                    content=f"seed-{i}",
                )
                for i in range(20)
            ]
        )
        mesh.embed_project("default")
        stop = threading.Event()
        errors: list[BaseException] = []

        def writer() -> None:
            try:
                for i in range(30):
                    Memory(db_path=self.db, embedder="fallback").store(f"w-{i}-{time.time()}")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                stop.set()

        def reader() -> None:
            try:
                while not stop.is_set():
                    Memory(db_path=self.db, embedder="fallback").query("seed", top_k=3)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
            self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(MemoryStore(self.db).database_status().foreign_keys)


class AiderParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.home = self.root / "home"
        self._home_ctx = temp_home(self.home)
        self._home_ctx.__enter__()
        self.chat = self.root / ".aider.chat.history.md"
        self.chat.write_text(
            (FIXTURES / "sample.aider.chat.history.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._home_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def test_roles_order_source_keys_and_stable_reparse(self) -> None:
        first = parse_aider_file("aider", "proj", self.chat)
        second = parse_aider_file("aider", "proj", self.chat)
        self.assertEqual([r.source_key for r in first], [r.source_key for r in second])
        self.assertEqual([r.timestamp for r in first], [r.timestamp for r in second])
        self.assertEqual(first[0].conversation_id, conversation_id_for_path(self.chat))
        roles = [r.role for r in first]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        self.assertIn("system", roles)  # tool/status mapped
        # Identical user text in different turns → different source keys.
        users = [r for r in first if r.role == "user" and "Please also add a unit test" in r.content]
        self.assertEqual(len(users), 2)
        self.assertNotEqual(users[0].source_key, users[1].source_key)
        # Session header timestamps (evidenced).
        for r in first:
            self.assertTrue(r.timestamp.startswith("2024-06-01"))
            meta = json.loads(r.metadata_json)
            self.assertEqual(meta["timestamp_origin"], "aider_session_header")
            self.assertFalse(meta["timestamp_synthetic"])
            # No invented subtypes.
            self.assertNotIn("subtype", meta)
            self.assertNotIn("assistant_subtype", meta)

    def test_code_fence_does_not_split_turns(self) -> None:
        records = parse_aider_file("aider", "proj", self.chat)
        assistants = [r for r in records if r.role == "assistant"]
        self.assertTrue(any("login():" in r.content for r in assistants))
        fence = self.root / "fence.md"
        fence.write_text(
            "# aider chat started at 2024-06-01 12:00:00\n\n"
            "#### ask\n\n"
            "Before\n```\n#### not a turn\n```\nAfter\n",
            encoding="utf-8",
        )
        fenced = parse_aider_file("aider", "proj", fence)
        users = [r for r in fenced if r.role == "user"]
        self.assertEqual(len(users), 1)
        assistants = [r for r in fenced if r.role == "assistant"]
        self.assertEqual(len(assistants), 1)
        self.assertIn("#### not a turn", assistants[0].content)

    def test_append_without_session_header_keeps_old_timestamps(self) -> None:
        path = self.root / "nosession.md"
        path.write_text(
            (FIXTURES / "no_session.aider.chat.history.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        # Force an early mtime, then parse.
        os.utime(path, (1_000_000, 1_000_000))
        first = parse_aider_file("aider", "proj", path)
        self.assertGreaterEqual(len(first), 2)
        old_keys = [r.source_key for r in first]
        old_ts = {r.source_key: r.timestamp for r in first}
        for r in first:
            meta = json.loads(r.metadata_json)
            self.assertEqual(meta["timestamp_origin"], "synthetic_epoch")
            self.assertTrue(meta["timestamp_synthetic"])

        db = self.root / "ns.db"
        mesh = MemoryMesh(Settings(db_path=db, embedding_backend="fallback"))
        mesh.init()
        n1 = mesh.ingest_file("aider", "proj", path)
        self.assertEqual(n1, len(first))

        # Append a turn and change mtime dramatically.
        path.write_text(
            path.read_text(encoding="utf-8") + "\n#### New appended\n\nAck new.\n",
            encoding="utf-8",
        )
        os.utime(path, (2_000_000_000, 2_000_000_000))
        second = parse_aider_file("aider", "proj", path)
        for key in old_keys:
            match = next(r for r in second if r.source_key == key)
            self.assertEqual(match.timestamp, old_ts[key])
        new_keys = [r.source_key for r in second if r.source_key not in old_keys]
        self.assertTrue(new_keys)
        n2 = mesh.ingest_file("aider", "proj", path)
        self.assertEqual(n2, len(new_keys))

    def test_append_only_reingest(self) -> None:
        db = self.root / "a.db"
        mesh = MemoryMesh(Settings(db_path=db, embedding_backend="fallback"))
        mesh.init()
        n1 = mesh.ingest_file("aider", "proj", self.chat)
        n2 = mesh.ingest_file("aider", "proj", self.chat)
        self.assertGreater(n1, 0)
        self.assertEqual(n2, 0)
        self.chat.write_text(
            self.chat.read_text(encoding="utf-8") + "\n#### Brand new appended turn\n\nAck.\n",
            encoding="utf-8",
        )
        n3 = mesh.ingest_file("aider", "proj", self.chat)
        self.assertEqual(n3, 2)  # new user + assistant

    def test_input_history_user_only(self) -> None:
        path = self.root / ".aider.input.history"
        path.write_text(
            (FIXTURES / "aider.input.history").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        records, _ = parse_aider_input_history("aider", "proj", path)
        self.assertTrue(records)
        self.assertTrue(all(r.role == "user" for r in records))
        self.assertEqual(len({r.source_key for r in records}), len(records))

    def test_unicode_and_malformed_isolation(self) -> None:
        path = self.root / "u.md"
        path.write_text(
            "# aider chat started at 2024-06-01 12:00:00\n\n#### café 日本語\n\nОтвет\n",
            encoding="utf-8",
        )
        records = parse_aider_file("aider", "proj", path)
        self.assertGreaterEqual(len(records), 2)
        self.assertIn("café", records[0].content)


class ProviderRegistryTests(unittest.TestCase):
    def test_registry_consistency(self) -> None:
        from deepiri_memorymesh.providers.registry import (
            allows_automatic_discovery,
            allows_native_integration,
            native_integration_provider_names,
            normalize_provider_key,
        )
        from deepiri_memorymesh.integrations import list_targets, install_native_integration

        natives = [c for c in list_providers() if c.parser_kind == "native"]
        mapping = native_parser_map()
        for cap in natives:
            self.assertTrue(cap.parser_symbol)
            self.assertIn(cap.name, mapping)
            self.assertTrue(cap.automatic_discovery)
            self.assertTrue(allows_automatic_discovery(cap.name))
        unsupported = [c for c in list_providers() if c.parser_kind == "unsupported"]
        self.assertTrue(any(c.name == "copilot" for c in unsupported))
        self.assertFalse(get_provider("openai").automatic_discovery)
        self.assertEqual(get_provider("jsonl").parser_kind, "generic-explicit")
        self.assertEqual(normalize_provider_key("json"), "jsonl")
        self.assertEqual(normalize_provider_key("anthropic"), "claude")

        # Integrations must track registry natives only.
        integ_names = {t.key for t in list_targets()}
        for name in native_integration_provider_names():
            self.assertIn(name, integ_names)
            self.assertTrue(allows_native_integration(name))
        for name in ("copilot", "openai", "cline"):
            self.assertFalse(allows_native_integration(name))
            self.assertFalse(allows_automatic_discovery(name))
            with self.assertRaises(ValueError):
                install_native_integration(name, "proj")

    def test_defaults_exclude_placeholders(self) -> None:
        settings = Settings()
        for name in ("copilot", "cline", "cody", "openai", "ollama_local"):
            self.assertNotIn(name, settings.providers)
            self.assertNotIn(name, settings.provider_paths)

    def test_old_yaml_loads_and_sync_auto_skips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            with temp_home(home):
                cfg = home / ".config" / "deepiri-memorymesh" / "config.yaml"
                cfg.parent.mkdir(parents=True)
                cfg.write_text(
                    f"db_path: {home / 'memory.db'}\nembedding_backend: fallback\n"
                    "providers: [claude, copilot, jsonl]\n"
                    "provider_paths:\n  claude: ~/.claude\n  copilot: ~/.copilot\n"
                    "provider_globs: {}\n",
                    encoding="utf-8",
                )
                settings = Settings.load()
                self.assertIn("copilot", settings.providers)
                mesh = MemoryMesh(settings)
                mesh.init()
                report = mesh.sync_auto_report("p")
                skipped = {o.provider: o for o in report.outcomes if o.skipped}
                self.assertIn("copilot", skipped)
                self.assertEqual(skipped["copilot"].classification, "unsupported")
                self.assertFalse(any(o.provider == "copilot" and not o.skipped for o in report.outcomes))


class CandidateRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.db = self.root / "q.db"
        self.home = self.root / "home"
        self._home_ctx = temp_home(self.home)
        self._home_ctx.__enter__()
        self.settings = Settings(
            db_path=self.db,
            embedding_backend="fallback",
            retrieval_mode="auto",
            retrieval_exact_threshold=5,
            retrieval_candidate_limit=8,
        )
        self.mesh = MemoryMesh(self.settings)
        self.mesh.init()

    def tearDown(self) -> None:
        self._home_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def test_tokenize_unicode_and_duplicates(self) -> None:
        terms = tokenize_for_index("Hello HELLO café café!!!")
        self.assertEqual(terms.count("hello"), 1)
        self.assertIn("café", terms)

    def test_inserts_index_terms_and_modes(self) -> None:
        mem = Memory(db_path=self.db, embedder="fallback")
        mem.store("azure cloud preference documented")
        with self.mesh.store.connection() as conn:
            status = search_index_status(conn)
            self.assertGreater(status.term_rows, 0)
        exact = self.mesh.query_with_report("default", "azure cloud", strategy="exact")
        self.assertEqual(exact.report.strategy_used, "exact")
        # Below threshold → auto uses exact.
        auto_small = self.mesh.query_with_report("default", "azure cloud", strategy="auto")
        self.assertEqual(auto_small.report.strategy_used, "exact")

    def test_indexed_bounded_and_no_silent_full_scan(self) -> None:
        # Create many embeddings above threshold.
        rows = []
        for i in range(40):
            content = f"document number {i} about topic-{i} unique-token-{i}"
            rows.append(
                MemoryRecord(
                    provider="python",
                    project="default",
                    conversation_id="python-api",
                    role="memory",
                    content=content,
                    source_key=f"sk-{i}",
                )
            )
        self.mesh.store.insert_messages(rows)
        self.mesh.embed_project("default")
        result = self.mesh.query_with_report(
            "default",
            "unique-token-7 topic",
            strategy="indexed",
            candidate_limit=5,
        )
        self.assertEqual(result.report.strategy_used, "indexed")
        self.assertLessEqual(result.report.candidate_message_count, 5)
        self.assertLessEqual(result.report.embeddings_scored, 5)
        self.assertGreater(result.report.total_eligible_embeddings, 5)
        # Indexed empty query tokens → diagnostic, not silent full scan.
        empty = self.mesh.query_with_report("default", "!!!", strategy="indexed")
        self.assertEqual(empty.report.strategy_used, "indexed")
        self.assertIsNotNone(empty.report.exact_fallback_reason)
        self.assertEqual(empty.rows, [])

    def test_thousands_bounded_scoring(self) -> None:
        n = 1200
        records = [
            MemoryRecord(
                provider="python",
                project="default",
                conversation_id="python-api",
                role="memory",
                content=f"row-{i} payload noise words alpha beta gamma {i}",
                source_key=f"bulk-{i}",
            )
            for i in range(n)
        ]
        # Insert in chunks.
        for start in range(0, n, 200):
            self.mesh.store.insert_messages(records[start : start + 200])
        self.mesh.embed_project("default")
        result = self.mesh.query_with_report(
            "default",
            "row-42 alpha",
            strategy="indexed",
            candidate_limit=32,
            top_k=5,
        )
        self.assertEqual(result.report.strategy_used, "indexed")
        self.assertLessEqual(result.report.embeddings_scored, 32)
        self.assertGreaterEqual(result.report.total_eligible_embeddings, n)

    def test_indexed_prefers_embedded_over_unembedded_lexical_matches(self) -> None:
        limit = 5
        # Many lexically matching messages WITHOUT embeddings.
        unembedded = [
            MemoryRecord(
                provider="python",
                project="default",
                conversation_id="python-api",
                role="memory",
                content=f"special-token filler {i}",
                source_key=f"noemb-{i}",
            )
            for i in range(limit + 3)
        ]
        self.mesh.store.insert_messages(unembedded)
        # One matching message WITH embedding.
        embedded = MemoryRecord(
            provider="python",
            project="default",
            conversation_id="python-api",
            role="memory",
            content="special-token unique-embedded-hit",
            source_key="emb-1",
        )
        mid = self.mesh.store.insert_message_returning_id(embedded)
        self.assertIsNotNone(mid)
        from deepiri_memorymesh.embeddings import Embedder

        emb = Embedder("fallback")
        self.mesh.store.save_embedding(int(mid), emb.dumps(emb.embed(embedded.content)))

        result = self.mesh.query_with_report(
            "default",
            "special-token",
            strategy="indexed",
            candidate_limit=limit,
            top_k=3,
        )
        self.assertEqual(result.report.strategy_used, "indexed")
        self.assertEqual(result.report.total_eligible_embeddings, 1)
        self.assertEqual(result.report.candidate_message_count, 1)
        self.assertLessEqual(result.report.embeddings_scored, 1)
        self.assertTrue(result.rows)
        self.assertIn("unique-embedded-hit", result.rows[0]["content"])
        # Indexed path must not silently escalate to full scan of unembedded rows.
        self.assertNotEqual(result.report.strategy_used, "exact")


class TuiServiceLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.home = self.root / "home"
        self._home_ctx = temp_home(self.home)
        self._home_ctx.__enter__()
        self.db = self.root / "tui.db"
        self.settings = Settings(db_path=self.db, embedding_backend="fallback")
        MemoryStore(self.db).init()

    def tearDown(self) -> None:
        self._home_ctx.__exit__(None, None, None)
        self._tmpdir.cleanup()

    def test_plain_tui_starts_no_service(self) -> None:
        from deepiri_memorymesh import cli as cli_mod
        from deepiri_memorymesh import supervised_service as ss

        with mock.patch.object(cli_mod, "run_tui") as run_tui:
            with mock.patch.object(ss, "SupervisedService") as sup_cls:
                with mock.patch.object(ss, "detect_existing_service") as detect:
                    detect.return_value = mock.Mock(ok=False, compatible=False)
                    # Typer wraps commands; call the underlying function.
                    fn = getattr(cli_mod.tui, "__wrapped__", cli_mod.tui)
                    try:
                        fn(project="p", with_service=False, host="127.0.0.1", port=8765)
                    except TypeError:
                        # Some typer versions expose .callback
                        cli_mod.tui(project="p", with_service=False, host="127.0.0.1", port=8765)
                run_tui.assert_called_once()
                # Plain path imports SupervisedService only when --with-service.
                # detect_existing_service may still run.
        source = Path(cli_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("start_new_session", source)
        self.assertNotIn("subprocess.Popen", source)

    def test_owned_service_starts_and_stops(self) -> None:
        svc = SupervisedService(host="127.0.0.1", port=0, settings=self.settings)
        # port 0 not ideal for health URL; bind ephemeral via temporary server pattern.
        # Use an explicit free port.
        probe = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        port = probe.server_address[1]
        probe.server_close()
        svc = SupervisedService(host="127.0.0.1", port=port, settings=self.settings)
        result = svc.start()
        self.assertEqual(result, "started")
        self.assertTrue(svc.owned)
        self.assertTrue(svc.thread and svc.thread.is_alive())
        health = detect_existing_service("127.0.0.1", port)
        self.assertTrue(health.ok and health.compatible)
        self.assertIsNotNone(health.identity)
        assert health.identity is not None
        self.assertTrue(health.identity.db_identity)
        self.assertNotIn("/", health.identity.db_identity)
        self.assertNotIn(str(self.db), json.dumps(health.raw))
        self.assertNotIn("db_path", health.raw)
        svc.shutdown()
        self.assertFalse(svc.owned)
        if svc.thread:
            self.assertFalse(svc.thread.is_alive())
        # Port reusable
        svc2 = SupervisedService(host="127.0.0.1", port=port, settings=self.settings)
        self.assertEqual(svc2.start(), "started")
        svc2.shutdown()

    def test_preexisting_not_stopped(self) -> None:
        probe = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        port = probe.server_address[1]
        probe.server_close()
        owner = SupervisedService(host="127.0.0.1", port=port, settings=self.settings)
        self.assertEqual(owner.start(), "started")
        guest = SupervisedService(host="127.0.0.1", port=port, settings=self.settings)
        self.assertIn(guest.start(), {"reused_existing", "reused_existing_after_race"})
        self.assertFalse(guest.owned)
        guest.shutdown()  # must not stop owner
        health = detect_existing_service("127.0.0.1", port)
        self.assertTrue(health.ok)
        owner.shutdown()

    def test_unrelated_process_not_killed(self) -> None:
        class _H(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                body = b'{"ok":true,"service":"other"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            svc = SupervisedService(host="127.0.0.1", port=port, settings=self.settings)
            self.assertEqual(svc.start(), "port_conflict_unrelated")
            self.assertFalse(svc.owned)
            # Unrelated still alive
            health = detect_existing_service("127.0.0.1", port)
            self.assertTrue(health.ok)
            self.assertFalse(health.compatible)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

    def test_owned_stops_after_exception(self) -> None:
        probe = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        port = probe.server_address[1]
        probe.server_close()
        svc = SupervisedService(host="127.0.0.1", port=port, settings=self.settings)
        try:
            with svc:
                self.assertEqual(svc.start_result, "started")
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertFalse(svc.owned)
        if svc.thread:
            self.assertFalse(svc.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
