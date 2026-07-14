"""Focused regression tests for T28/T30/T15/T14/T35."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import threading
import unittest
import warnings
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from deepiri_memorymesh import Memory
from deepiri_memorymesh.cli import app
from deepiri_memorymesh.config import Settings, normalize_db_path
from deepiri_memorymesh.embedding_codec import (
    BACKEND_HASH_V1,
    BACKEND_ST_MINILM,
    EmbeddingCodecError,
    EmbeddingIncompatibilityError,
    REEMBED_HINT,
    cosine_strict,
    parse_embedding,
    sanitize_bridge_diagnostic,
    serialize_embedding,
)
from deepiri_memorymesh.embeddings import Embedder
from deepiri_memorymesh.file_scan import collect_provider_files
from deepiri_memorymesh.integrations import (
    install_bridge_script,
    install_hook_script,
    install_native_integration,
)
from deepiri_memorymesh.models import MemoryRecord
from deepiri_memorymesh.retrieval import format_rank_diagnostic, rank_rows_with_report
from deepiri_memorymesh.sync_service import MemoryMesh


class DbPathNormalizationTests(unittest.TestCase):
    """T28."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.home = Path(self._tmpdir.name) / "home"
        self.home.mkdir()
        self.cwd = Path(self._tmpdir.name) / "cwd"
        self.cwd.mkdir()
        self._home_patch = mock.patch.dict(os.environ, {"HOME": str(self.home)}, clear=False)
        self._home_patch.start()
        self._prev_cwd = Path.cwd()
        os.chdir(self.cwd)

    def tearDown(self) -> None:
        os.chdir(self._prev_cwd)
        self._home_patch.stop()
        self._tmpdir.cleanup()

    def test_tilde_expands_under_temp_home(self) -> None:
        path = normalize_db_path("~/mesh.db")
        self.assertEqual(path, self.home / "mesh.db")
        self.assertFalse((Path.cwd() / "~").exists())

    def test_nested_config_tilde_expands(self) -> None:
        path = normalize_db_path("~/.config/deepiri-memorymesh/custom.db")
        self.assertEqual(
            path,
            self.home / ".config" / "deepiri-memorymesh" / "custom.db",
        )

    def test_absolute_path_unchanged(self) -> None:
        absolute = self.home / "abs.db"
        path = normalize_db_path(str(absolute))
        self.assertEqual(path, absolute)

    def test_relative_path_stays_cwd_relative(self) -> None:
        path = normalize_db_path("relative/mesh.db")
        self.assertFalse(path.is_absolute())
        self.assertEqual(path, Path("relative/mesh.db"))
        # Resolves against process CWD, not config directory.
        self.assertEqual((Path.cwd() / path).resolve(), (self.cwd / "relative" / "mesh.db").resolve())

    def test_missing_db_does_not_fail_on_load(self) -> None:
        cfg_dir = self.home / ".config" / "deepiri-memorymesh"
        cfg_dir.mkdir(parents=True)
        cfg_path = cfg_dir / "config.yaml"
        cfg_path.write_text(
            "db_path: ~/.config/deepiri-memorymesh/custom.db\n"
            "embedding_backend: fallback\n",
            encoding="utf-8",
        )
        before = cfg_path.read_text(encoding="utf-8")
        settings = Settings.load(cfg_path)
        after = cfg_path.read_text(encoding="utf-8")
        self.assertEqual(before, after)
        expected = self.home / ".config" / "deepiri-memorymesh" / "custom.db"
        self.assertEqual(settings.db_path, expected)
        self.assertFalse(expected.exists())
        # Opening the store creates parents/db; config YAML still untouched.
        mesh = MemoryMesh(settings)
        mesh.init()
        self.assertTrue(expected.exists())
        self.assertEqual(cfg_path.read_text(encoding="utf-8"), before)
        self.assertFalse((self.cwd / "~").exists())
        self.assertFalse((Path.cwd() / "~").exists())


class EmbeddingFallbackStatusTests(unittest.TestCase):
    """T30."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_explicit_fallback_no_failure_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            emb = Embedder("fallback")
            status = emb.status()
        fallback_warns = [
            w for w in caught if "fallback" in str(w.message).lower() and "requested=" in str(w.message)
        ]
        self.assertEqual(fallback_warns, [])
        self.assertEqual(status.requested_backend, "fallback")
        self.assertEqual(status.active_backend, "fallback")
        self.assertFalse(status.fallback_occurred)
        self.assertIsNone(status.fallback_reason)
        self.assertEqual(status.stable_backend_id, BACKEND_HASH_V1)

    def test_sentence_transformers_import_failure_warns_once(self) -> None:
        import sys
        import types

        fake = types.ModuleType("sentence_transformers")

        class _Boom:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("model load failed for test")

        fake.SentenceTransformer = _Boom  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                emb = Embedder("sentence-transformers")
                emb._emit_fallback_warning()
                emb._emit_fallback_warning()
        msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertEqual(len(msgs), 1)
        self.assertIn("requested='sentence-transformers'", msgs[0])
        self.assertIn("active='fallback'", msgs[0])
        self.assertIn("retrieval quality may differ", msgs[0])
        self.assertNotIn("secret memory text", msgs[0])
        # stacklevel should attribute the warning to this test module, not embeddings.py
        warn_files = [
            w.filename for w in caught if issubclass(w.category, UserWarning)
        ]
        self.assertTrue(any("test_batch3_fixes" in f for f in warn_files), warn_files)
        status = emb.status()
        self.assertEqual(status.requested_backend, "sentence-transformers")
        self.assertEqual(status.active_backend, "fallback")
        self.assertTrue(status.fallback_occurred)
        self.assertIn("RuntimeError", status.fallback_reason or "")

    def test_mocked_sentence_transformers_no_fallback_warning(self) -> None:
        import sys
        import types

        fake = types.ModuleType("sentence_transformers")

        class _Vec:
            def tolist(self):
                return [0.1] * 384

        class _FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, texts, normalize_embeddings=True):
                return [_Vec()]

        fake.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                emb = Embedder("sentence-transformers")
                _ = emb.embed("hello")
        fallback_warns = [w for w in caught if "Embedding backend fallback" in str(w.message)]
        self.assertEqual(fallback_warns, [])
        status = emb.status()
        self.assertEqual(status.active_backend, "sentence-transformers")
        self.assertFalse(status.fallback_occurred)

    def test_unsupported_backend_fails(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Embedder("mystery-backend")
        self.assertIn("Unsupported embedding backend", str(ctx.exception))

    def test_simple_memory_auto_reports_fallback(self) -> None:
        import sys
        import types

        fake = types.ModuleType("sentence_transformers")

        class _Boom:
            def __init__(self, *args, **kwargs):
                raise ImportError("no st")

        fake.SentenceTransformer = _Boom  # type: ignore[attr-defined]
        db = self.root / "m.db"
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mem = Memory(db_path=db, embedder="auto")
        self.assertTrue(any("Embedding backend fallback" in str(w.message) for w in caught))
        status = mem.embedding_status()
        self.assertEqual(status["requested_backend"], "auto")
        self.assertEqual(status["active_backend"], "fallback")
        self.assertTrue(status["fallback_occurred"])

    def test_cli_embedding_status_shows_active(self) -> None:
        db = self.root / "cli.db"
        settings = Settings(db_path=db, embedding_backend="fallback")
        mesh = MemoryMesh(settings)
        mesh.init()
        runner = CliRunner()
        with mock.patch("deepiri_memorymesh.cli._mesh", return_value=mesh):
            result = runner.invoke(app, ["embedding-status"])
        self.assertEqual(result.exit_code, 0, result.stdout + result.stderr)
        self.assertIn("active_backend=fallback", result.stdout)
        self.assertIn("requested_backend=fallback", result.stdout)
        self.assertIn("fallback_occurred=false", result.stdout)


class EmbeddingCompatibilityTests(unittest.TestCase):
    """T15."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        import deepiri_memorymesh.embedding_codec as codec

        codec._legacy_compat_warned = False

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_versioned_fallback_round_trip(self) -> None:
        vec = [0.0] * 128
        vec[0] = 1.0
        raw = serialize_embedding(vec, backend="fallback", model=BACKEND_HASH_V1)
        parsed = parse_embedding(raw)
        self.assertEqual(parsed.backend, BACKEND_HASH_V1)
        self.assertEqual(parsed.model, BACKEND_HASH_V1)
        self.assertEqual(parsed.dimensions, 128)
        self.assertEqual(parsed.version, 1)
        self.assertFalse(parsed.legacy)
        self.assertEqual(parsed.vector[0], 1.0)

    def test_legacy_raw_array_parseable(self) -> None:
        parsed = parse_embedding("[1.0, 0.0, 0.0]")
        self.assertTrue(parsed.legacy)
        self.assertIsNone(parsed.backend)
        self.assertEqual(parsed.dimensions, 3)

    def test_equal_dimension_compatible_scores(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_strict(a, b), 1.0)

    def test_dimension_mismatch_rejected(self) -> None:
        short = [0.0] * 128
        long = [0.0] * 384
        with self.assertRaises(EmbeddingIncompatibilityError):
            cosine_strict(short, long)
        with self.assertRaises(EmbeddingIncompatibilityError):
            cosine_strict(long, short)

    def test_serialize_rejects_backend_dimension_mismatch(self) -> None:
        with self.assertRaises(EmbeddingCodecError):
            serialize_embedding([0.0] * 128, backend=BACKEND_ST_MINILM, model=BACKEND_ST_MINILM)
        with self.assertRaises(EmbeddingCodecError):
            serialize_embedding([0.0] * 384, backend="fallback", model=BACKEND_HASH_V1)
        with self.assertRaises(EmbeddingCodecError):
            serialize_embedding(
                [0.0] * 128, backend="fallback", model=BACKEND_ST_MINILM
            )
        with self.assertRaises(EmbeddingCodecError):
            serialize_embedding([True, False] + [0.0] * 126, backend="fallback")

    def test_rank_skips_mixed_dimensions_and_malformed(self) -> None:
        query = [1.0] + [0.0] * 127
        q_meta = parse_embedding(
            serialize_embedding(query, backend="fallback", model=BACKEND_HASH_V1)
        )
        good = serialize_embedding(query, backend="fallback", model=BACKEND_HASH_V1)
        wrong_dim = serialize_embedding(
            [0.0] * 384, backend=BACKEND_ST_MINILM, model=BACKEND_ST_MINILM
        )
        # Corrupt MiniLM envelope with wrong dimensions — parse must fail closed.
        wrong_backend = json.dumps(
            {
                "version": 1,
                "backend": BACKEND_ST_MINILM,
                "model": BACKEND_ST_MINILM,
                "dimensions": 128,
                "vector": [0.0] * 128,
            }
        )
        legacy = json.dumps([1.0] + [0.0] * 127)
        version_garbage = json.dumps(
            {
                "version": "garbage",
                "backend": BACKEND_HASH_V1,
                "dimensions": 128,
                "vector": [0.0] * 128,
            }
        )
        future_version = json.dumps(
            {
                "version": 99,
                "backend": BACKEND_HASH_V1,
                "dimensions": 128,
                "vector": [0.0] * 128,
            }
        )
        rows = [
            {"message_id": 1, "content": "good", "provider": "p", "conversation_id": "c", "embedding_json": good},
            {"message_id": 2, "content": "dim", "provider": "p", "conversation_id": "c", "embedding_json": wrong_dim},
            {
                "message_id": 3,
                "content": "backend",
                "provider": "p",
                "conversation_id": "c",
                "embedding_json": wrong_backend,
            },
            {"message_id": 4, "content": "legacy", "provider": "p", "conversation_id": "c", "embedding_json": legacy},
            {"message_id": 5, "content": "bad", "provider": "p", "conversation_id": "c", "embedding_json": "{not-json"},
            {
                "message_id": 6,
                "content": "missing",
                "provider": "p",
                "conversation_id": "c",
                "embedding_json": json.dumps({"version": 1, "backend": BACKEND_HASH_V1}),
            },
            {
                "message_id": 7,
                "content": "nonnum",
                "provider": "p",
                "conversation_id": "c",
                "embedding_json": json.dumps(
                    {
                        "version": 1,
                        "backend": BACKEND_HASH_V1,
                        "dimensions": 128,
                        "vector": ["x"] + [0.0] * 127,
                    }
                ),
            },
            {
                "message_id": 8,
                "content": "dimdecl",
                "provider": "p",
                "conversation_id": "c",
                "embedding_json": json.dumps(
                    {
                        "version": 1,
                        "backend": BACKEND_HASH_V1,
                        "dimensions": 3,
                        "vector": [1.0, 0.0],
                    }
                ),
            },
            {
                "message_id": 9,
                "content": "verstr",
                "provider": "p",
                "conversation_id": "c",
                "embedding_json": version_garbage,
            },
            {
                "message_id": 10,
                "content": "verfuture",
                "provider": "p",
                "conversation_id": "c",
                "embedding_json": future_version,
            },
        ]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ranked, report = rank_rows_with_report(query, rows, top_k=8, query_meta=q_meta)
        contents = {r["content"] for r in ranked}
        self.assertIn("good", contents)
        self.assertIn("legacy", contents)
        self.assertNotIn("dim", contents)
        self.assertNotIn("backend", contents)
        self.assertNotIn("verstr", contents)
        self.assertNotIn("verfuture", contents)
        self.assertGreaterEqual(report.skipped_malformed, 5)
        self.assertGreaterEqual(report.skipped_incompatible, 1)
        self.assertGreaterEqual(report.legacy_compatible, 1)
        self.assertEqual(report.scored_versioned, 1)
        diag = format_rank_diagnostic(report)
        assert diag is not None
        self.assertIn("legacy", diag)
        self.assertIn("incompatible", diag)
        self.assertIn("malformed", diag)
        self.assertIn("re-embed", diag.lower())

    def test_unknown_query_backend_never_scored_versioned(self) -> None:
        query = [1.0] + [0.0] * 127
        versioned = serialize_embedding(query, backend="fallback", model=BACKEND_HASH_V1)
        legacy = json.dumps([0.5] + [0.0] * 127)
        rows = [
            {
                "message_id": 1,
                "content": "versioned",
                "provider": "p",
                "conversation_id": "c",
                "embedding_json": versioned,
            },
            {
                "message_id": 2,
                "content": "legacy",
                "provider": "p",
                "conversation_id": "c",
                "embedding_json": legacy,
            },
        ]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            ranked, report = rank_rows_with_report(query, rows, top_k=8)
        self.assertEqual(report.scored_versioned, 0)
        self.assertGreaterEqual(report.skipped_incompatible, 1)
        self.assertGreaterEqual(report.legacy_compatible, 1)
        contents = {r["content"] for r in ranked}
        self.assertIn("legacy", contents)
        self.assertNotIn("versioned", contents)

    def test_parse_embedding_schema_strict(self) -> None:
        vec = [0.0] * 128
        base = {
            "version": 1,
            "backend": BACKEND_HASH_V1,
            "model": BACKEND_HASH_V1,
            "dimensions": 128,
            "vector": vec,
        }
        # Happy path
        parsed = parse_embedding(json.dumps(base))
        self.assertEqual(parsed.backend, BACKEND_HASH_V1)

        cases = [
            {k: v for k, v in base.items() if k != "version"},  # missing version
            {**base, "version": "1"},  # string version
            {**base, "version": 1.5},  # fractional
            {**base, "version": 2},  # unsupported future
            {k: v for k, v in base.items() if k != "backend"},  # missing backend
            {**base, "backend": "unknown-backend"},
            {k: v for k, v in base.items() if k != "dimensions"},  # missing dimensions
            {**base, "dimensions": 384},  # incorrect backend dimension
            {**base, "model": BACKEND_ST_MINILM},  # contradictory model
            {**base, "vector": [True] + [0.0] * 127},  # bool vector values
            {**base, "dimensions": True},  # bool dimensions
        ]
        for payload in cases:
            with self.assertRaises(EmbeddingCodecError):
                parse_embedding(json.dumps(payload))

        with self.assertRaises(EmbeddingCodecError):
            parse_embedding(b"\xff\xfe not utf-8")
        with self.assertRaises(EmbeddingCodecError):
            parse_embedding("{")
        with self.assertRaises(EmbeddingCodecError):
            parse_embedding(json.dumps({"version": 1, "backend": BACKEND_HASH_V1}))

    def test_platform_reembed_overwrites_with_versioned(self) -> None:
        db = self.root / "mesh.db"
        mesh = MemoryMesh(Settings(db_path=db, embedding_backend="fallback"))
        mesh.init()
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="cursor",
                    project="p",
                    conversation_id="c1",
                    role="user",
                    content="reembed-me",
                )
            ]
        )
        msg_id = int(mesh.store.list_messages("p")[0]["id"])
        mesh.store.save_embedding(msg_id, json.dumps([1.0] + [0.0] * 127))
        mesh.embed_project("p")
        raw = mesh.store.list_embeddings("p")[0]["embedding_json"]
        parsed = parse_embedding(raw)
        self.assertFalse(parsed.legacy)
        self.assertEqual(parsed.backend, BACKEND_HASH_V1)
        self.assertEqual(parsed.dimensions, 128)

    def test_simple_memory_versioned_and_strict_query(self) -> None:
        db = self.root / "simple.db"
        mem = Memory(db_path=db, embedder="fallback")
        mem.store("alpha memory")
        conn = sqlite3.connect(db)
        emb = conn.execute(
            """
            SELECT e.embedding_json
            FROM memory_embeddings e
            JOIN memory_messages m ON e.message_id = m.id
            WHERE m.content = ?
            """,
            ("alpha memory",),
        ).fetchone()[0]
        conn.close()
        parsed = parse_embedding(emb)
        self.assertEqual(parsed.backend, BACKEND_HASH_V1)
        self.assertFalse(parsed.legacy)

        # Inject incompatible legacy longer vector + compatible legacy into
        # the facade namespace (platform schema).
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA foreign_keys = ON")
        for content, vec in (
            ("legacy-ok", [1.0] + [0.0] * 127),
            ("legacy-bad-dim", [0.1] * 384),
        ):
            cur = conn.execute(
                """
                INSERT INTO memory_messages
                (provider, project, conversation_id, role, content, timestamp, metadata_json)
                VALUES ('python', 'default', 'python-api', 'memory', ?, 't', '{}')
                """,
                (content,),
            )
            conn.execute(
                "INSERT INTO memory_embeddings (message_id, embedding_json) VALUES (?, ?)",
                (cur.lastrowid, json.dumps(vec)),
            )
        conn.commit()
        conn.close()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = mem.query("alpha")
        self.assertIn("alpha memory", results)
        self.assertNotIn("legacy-bad-dim", results)
        self.assertIsNotNone(mem.last_query_report)
        self.assertGreaterEqual(mem.last_query_report.skipped, 1)
        self.assertIn("re-embed", REEMBED_HINT.lower())

    def test_malformed_helpers(self) -> None:
        with self.assertRaises(EmbeddingCodecError):
            parse_embedding("{")
        with self.assertRaises(EmbeddingCodecError):
            parse_embedding(json.dumps({"version": 1, "backend": BACKEND_HASH_V1}))
        with self.assertRaises(EmbeddingCodecError):
            parse_embedding(
                json.dumps(
                    {
                        "version": 1,
                        "backend": BACKEND_HASH_V1,
                        "dimensions": 128,
                        "vector": ["a"] + [0.0] * 127,
                    }
                )
            )
        with self.assertRaises(EmbeddingCodecError):
            parse_embedding(
                json.dumps(
                    {
                        "version": 1,
                        "backend": BACKEND_HASH_V1,
                        "dimensions": 9,
                        "vector": [1.0, 0.0],
                    }
                )
            )


class ProviderGlobCollectionTests(unittest.TestCase):
    """T14."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        (self.root / "root.json").write_text("{}", encoding="utf-8")
        (self.root / "root.jsonl").write_text("{}\n", encoding="utf-8")
        nested = self.root / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "nested.json").write_text("{}", encoding="utf-8")
        (nested / "deeper.jsonl").write_text("{}\n", encoding="utf-8")
        (self.root / "skip.txt").write_text("x", encoding="utf-8")
        (self.root / "dir.json").mkdir()
        # Duplicate match across patterns: same file via *.json and **/*.json

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _names(self, paths: list[Path]) -> list[str]:
        return sorted(str(p.relative_to(self.root)) for p in paths)

    def test_recursive_star_json(self) -> None:
        files = collect_provider_files(self.root, ["*.json"], recursive=True)
        names = self._names(files)
        self.assertEqual(names, ["a/b/nested.json", "root.json"])
        self.assertTrue(all(p.is_file() for p in files))

    def test_recursive_double_star_json(self) -> None:
        files = collect_provider_files(self.root, ["**/*.json"], recursive=True)
        self.assertEqual(self._names(files), ["a/b/nested.json", "root.json"])

    def test_recursive_multiple_patterns_dedupe_sorted(self) -> None:
        files = collect_provider_files(
            self.root, ["*.json", "**/*.json", "**/*.jsonl"], recursive=True
        )
        names = self._names(files)
        self.assertEqual(
            names,
            ["a/b/deeper.jsonl", "a/b/nested.json", "root.json", "root.jsonl"],
        )
        self.assertEqual(len(files), len(set(files)))

    def test_nonrecursive_star_json(self) -> None:
        files = collect_provider_files(self.root, ["*.json"], recursive=False)
        self.assertEqual(self._names(files), ["root.json"])

    def test_nonrecursive_double_star_normalized(self) -> None:
        files = collect_provider_files(self.root, ["**/*.json"], recursive=False)
        self.assertEqual(self._names(files), ["root.json"])

    def test_explicit_subdir_pattern(self) -> None:
        files = collect_provider_files(self.root, ["a/b/*.json"], recursive=True)
        self.assertEqual(self._names(files), ["a/b/nested.json"])

    def test_nonrecursive_subdir_pattern_does_not_traverse(self) -> None:
        files = collect_provider_files(self.root, ["a/b/*.json"], recursive=False)
        self.assertEqual(files, [])
        files2 = collect_provider_files(self.root, ["subdir/*.json"], recursive=False)
        self.assertEqual(files2, [])

    def test_symlink_directory_not_traversed(self) -> None:
        outside = self.root.parent / "outside_tree"
        outside.mkdir(exist_ok=True)
        (outside / "via-link.json").write_text("{}", encoding="utf-8")
        link = self.root / "symdir"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        recursive = collect_provider_files(self.root, ["*.json"], recursive=True)
        names = self._names(recursive)
        self.assertNotIn("symdir/via-link.json", names)
        nonrec = collect_provider_files(self.root, ["*.json"], recursive=False)
        self.assertEqual(self._names(nonrec), ["root.json"])

    def test_directories_excluded(self) -> None:
        files = collect_provider_files(self.root, ["*.json", "dir.json"], recursive=False)
        self.assertEqual(self._names(files), ["root.json"])

    def test_provider_defaults_via_sync(self) -> None:
        db = self.root / "db.sqlite"
        mesh = MemoryMesh(Settings(db_path=db, embedding_backend="fallback"))
        mesh.init()
        settings = Settings()
        globs = settings.provider_globs["cursor"]
        # Place a cursor-like chat file nested.
        chat = self.root / "proj" / "chat-session.jsonl"
        chat.parent.mkdir(parents=True)
        chat.write_text(json.dumps({"role": "user", "content": "hi"}) + "\n", encoding="utf-8")
        report = mesh.sync_directory_report(
            "cursor", "p", self.root / "proj", recursive=True, include_globs=globs
        )
        self.assertEqual(report.processed, 1)
        self.assertGreaterEqual(report.inserted, 1)


class HookBridgeTransferFailureTests(unittest.TestCase):
    """T35."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.home = Path(self._tmpdir.name)
        self.bin_dir = self.home / ".local" / "bin"
        self.bin_dir.mkdir(parents=True)
        self.capture_body = self.home / "captured.json"
        self.capture_url = self.home / "captured.url"
        self._home_patch = mock.patch.dict(os.environ, {"HOME": str(self.home)}, clear=False)
        self._home_patch.start()

    def tearDown(self) -> None:
        self._home_patch.stop()
        self._tmpdir.cleanup()

    def _clean_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"MEMORYMESH_PROJECT", "MEMORYMESH_URL", "MEMORYMESH_STRICT"}
        }
        env["HOME"] = str(self.home)
        env["PATH"] = f"{self.bin_dir}:{env.get('PATH', '')}"
        if extra:
            env.update(extra)
        return env

    def _install_fake_curl(self, *, fail: bool = False) -> None:
        curl = self.bin_dir / "curl"
        curl.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f'BODY_OUT="{self.capture_body}"\n'
            f'URL_OUT="{self.capture_url}"\n'
            f"FAIL={1 if fail else 0}\n"
            'url=""\n'
            'while [ "$#" -gt 0 ]; do\n'
            '  case "$1" in\n'
            "    --data-binary)\n"
            "      shift\n"
            '      if [ "${1:-}" = "@-" ]; then cat > "$BODY_OUT"; fi\n'
            "      shift || true\n"
            "      ;;\n"
            "    -X|-H) shift 2 || true ;;\n"
            "    -sS|--fail) shift ;;\n"
            '    http*|HTTPS*) url="$1"; shift ;;\n'
            "    *) shift ;;\n"
            "  esac\n"
            "done\n"
            'printf "%s" "$url" > "$URL_OUT"\n'
            'if [ "$FAIL" -eq 1 ]; then\n'
            '  echo "curl: simulated failure" >&2\n'
            "  exit 22\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        curl.chmod(0o755)

    def test_bridge_success_prints_ok(self) -> None:
        self._install_fake_curl(fail=False)
        bridge = install_bridge_script("cursor", "proj")
        target = self.home / "ok.json"
        target.write_text("{}", encoding="utf-8")
        proc = subprocess.run(
            ["bash", str(bridge), str(target)],
            check=False,
            capture_output=True,
            text=True,
            env=self._clean_env(),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("sync ok", proc.stdout)
        self.assertIn("--fail", bridge.read_text(encoding="utf-8"))
        self.assertNotIn("--fail-with-body", bridge.read_text(encoding="utf-8"))

    def test_bridge_failure_nonzero_no_sync_ok(self) -> None:
        self._install_fake_curl(fail=True)
        bridge = install_bridge_script("cursor", "proj")
        target = self.home / "bad.json"
        target.write_text("{}", encoding="utf-8")
        proc = subprocess.run(
            ["bash", str(bridge), str(target)],
            check=False,
            capture_output=True,
            text=True,
            env=self._clean_env(),
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("ingest failed", proc.stderr)
        self.assertNotIn("sync ok", proc.stdout)
        self.assertNotIn("{}", proc.stderr)

    def test_hook_soft_and_strict_failure(self) -> None:
        self._install_fake_curl(fail=True)
        hook = install_hook_script("gemini", "proj")
        export = self.home / "export.jsonl"
        export.write_text("{}\n", encoding="utf-8")
        soft = subprocess.run(
            ["bash", str(hook)],
            input=json.dumps({"transcript_path": str(export)}),
            check=False,
            capture_output=True,
            text=True,
            env=self._clean_env(),
        )
        self.assertEqual(soft.returncode, 0, soft.stderr)
        self.assertIn("ingest failed", soft.stderr)
        self.assertNotIn("|| true", hook.read_text(encoding="utf-8"))

        strict = subprocess.run(
            ["bash", str(hook)],
            input=json.dumps({"transcript_path": str(export)}),
            check=False,
            capture_output=True,
            text=True,
            env=self._clean_env({"MEMORYMESH_STRICT": "1"}),
        )
        self.assertNotEqual(strict.returncode, 0)
        self.assertIn("ingest failed", strict.stderr)

    def test_hook_success(self) -> None:
        self._install_fake_curl(fail=False)
        hook = install_hook_script("gemini", "proj")
        export = self.home / "export.jsonl"
        export.write_text("{}\n", encoding="utf-8")
        proc = subprocess.run(
            ["bash", str(hook)],
            input=json.dumps({"transcript_path": str(export)}),
            check=False,
            capture_output=True,
            text=True,
            env=self._clean_env(),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(self.capture_body.exists())

    def test_aider_exit_code_policies(self) -> None:
        paths = install_native_integration("aider", "proj")
        wrapper = next(p for p in paths if p.name == "aider-memorymesh")
        text = wrapper.read_text(encoding="utf-8")
        self.assertNotIn("|| true", text)
        self.assertIn("AIDER_RC", text)
        self.assertIn("MEMORYMESH_STRICT", text)

        aider = self.bin_dir / "aider"
        mm = self.bin_dir / "memorymesh"

        # Aider success + ingest success
        aider.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        aider.chmod(0o755)
        mm.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        mm.chmod(0o755)
        ok = subprocess.run(
            ["bash", str(wrapper)],
            check=False,
            capture_output=True,
            text=True,
            env=self._clean_env(),
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)

        # Aider success + ingest failure, soft
        mm.write_text("#!/usr/bin/env bash\necho fail >&2\nexit 7\n", encoding="utf-8")
        soft = subprocess.run(
            ["bash", str(wrapper)],
            check=False,
            capture_output=True,
            text=True,
            env=self._clean_env(),
        )
        self.assertEqual(soft.returncode, 0)
        self.assertIn("ingest failed", soft.stderr)

        # Aider success + ingest failure, strict
        strict = subprocess.run(
            ["bash", str(wrapper)],
            check=False,
            capture_output=True,
            text=True,
            env=self._clean_env({"MEMORYMESH_STRICT": "1"}),
        )
        self.assertNotEqual(strict.returncode, 0)

        # Aider failure preserved even if ingest would succeed / fail
        aider.write_text("#!/usr/bin/env bash\nexit 9\n", encoding="utf-8")
        mm.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        failed = subprocess.run(
            ["bash", str(wrapper)],
            check=False,
            capture_output=True,
            text=True,
            env=self._clean_env({"MEMORYMESH_STRICT": "1"}),
        )
        self.assertEqual(failed.returncode, 9)

    def test_transfer_push_reporting(self) -> None:
        db = self.home / "mesh.db"
        mesh = MemoryMesh(Settings(db_path=db, embedding_backend="fallback"))
        mesh.init()
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="claude",
                    project="p",
                    conversation_id="c",
                    role="user",
                    content="hello",
                )
            ]
        )

        # Missing bridge
        with self.assertLogs("deepiri_memorymesh.sync_service", level="WARNING"):
            path, count, push = mesh.transfer_with_report(
                "p", "claude", "cursor", push_via_bridge=True
            )
        self.assertGreaterEqual(count, 1)
        self.assertTrue(path.exists())
        self.assertTrue(push.attempted)
        self.assertFalse(push.success)
        self.assertIn("bridge not found", push.message)

        # Failing bridge
        bridge = self.bin_dir / "memorymesh-bridge-cursor"
        bridge.write_text("#!/usr/bin/env bash\necho bridge-boom >&2\nexit 3\n", encoding="utf-8")
        bridge.chmod(0o755)
        with self.assertLogs("deepiri_memorymesh.sync_service", level="WARNING"):
            _path, _count, push = mesh.transfer_with_report(
                "p", "claude", "cursor", push_via_bridge=True
            )
        self.assertFalse(push.success)
        self.assertEqual(push.returncode, 3)
        self.assertIn("bridge-boom", push.message)

        # Successful bridge
        bridge.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        bridge.chmod(0o755)
        _path, _count, push = mesh.transfer_with_report(
            "p", "claude", "cursor", push_via_bridge=True
        )
        self.assertTrue(push.success)

    def test_transfer_stderr_is_sanitized_and_bounded(self) -> None:
        db = self.home / "mesh-stderr.db"
        mesh = MemoryMesh(Settings(db_path=db, embedding_backend="fallback"))
        mesh.init()
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="claude",
                    project="p",
                    conversation_id="c",
                    role="user",
                    content="hello",
                )
            ]
        )
        secret = "SECRET_TRANSCRIPT_BODY_" + ("x" * 5000)
        bridge = self.bin_dir / "memorymesh-bridge-cursor"
        bridge.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' 'line1' '{secret}' $'ctl\\x01\\x02chars' 'tail' >&2\n"
            "exit 9\n",
            encoding="utf-8",
        )
        bridge.chmod(0o755)
        with self.assertLogs("deepiri_memorymesh.sync_service", level="WARNING") as cm:
            _path, _count, push = mesh.transfer_with_report(
                "p", "claude", "cursor", push_via_bridge=True
            )
        self.assertFalse(push.success)
        self.assertEqual(push.returncode, 9)
        self.assertLessEqual(len(push.message), 170)
        self.assertEqual(push.message, "tail")
        self.assertNotIn(secret, push.message)
        self.assertNotIn("SECRET_TRANSCRIPT", push.message)
        self.assertNotIn("\x01", push.message)
        joined = "\n".join(cm.output)
        self.assertNotIn(secret, joined)
        # Helper matches the same policy used for reports.
        long_line = "VISIBLE_PREFIX_" + ("y" * 1000)
        sanitized = sanitize_bridge_diagnostic(secret + "\n" + long_line)
        self.assertTrue(sanitized.startswith("VISIBLE_PREFIX_"))
        self.assertLessEqual(len(sanitized), 170)
        self.assertTrue(sanitized.endswith("..."))
        self.assertNotIn(secret, sanitized)

    def test_transfer_reports_are_per_call_under_concurrency(self) -> None:
        db = self.home / "mesh-conc.db"
        mesh = MemoryMesh(Settings(db_path=db, embedding_backend="fallback"))
        mesh.init()
        mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="claude",
                    project="p",
                    conversation_id="c",
                    role="user",
                    content="hello",
                )
            ]
        )
        slow = self.bin_dir / "memorymesh-bridge-cursor"
        fast = self.bin_dir / "memorymesh-bridge-gemini"
        slow.write_text(
            "#!/usr/bin/env bash\nsleep 0.25\necho slow-ok\nexit 0\n",
            encoding="utf-8",
        )
        slow.chmod(0o755)
        fast.write_text(
            "#!/usr/bin/env bash\necho fast-fail >&2\nexit 7\n",
            encoding="utf-8",
        )
        fast.chmod(0o755)

        results: dict[str, object] = {}
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def worker(name: str, target: str, expect_fail: bool) -> None:
            try:
                barrier.wait(timeout=5)
                if expect_fail:
                    with self.assertLogs("deepiri_memorymesh.sync_service", level="WARNING"):
                        _path, _count, push = mesh.transfer_with_report(
                            "p",
                            "claude",
                            target,
                            out_path=self.home / f"xfer-{name}.json",
                            push_via_bridge=True,
                        )
                else:
                    _path, _count, push = mesh.transfer_with_report(
                        "p",
                        "claude",
                        target,
                        out_path=self.home / f"xfer-{name}.json",
                        push_via_bridge=True,
                    )
                results[name] = push
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=("slow", "cursor", False))
        t2 = threading.Thread(target=worker, args=("fast", "gemini", True))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertFalse(t1.is_alive())
        self.assertFalse(t2.is_alive())
        slow_push = results["slow"]
        fast_push = results["fast"]
        self.assertTrue(slow_push.success)  # type: ignore[union-attr]
        self.assertFalse(fast_push.success)  # type: ignore[union-attr]
        self.assertEqual(fast_push.returncode, 7)  # type: ignore[union-attr]
        # Shared last_transfer_push may be either call; per-call reports stay correct.
        self.assertIn(mesh.last_transfer_push, (slow_push, fast_push))


class ConcurrentQueryReportHttpTests(unittest.TestCase):
    """Threaded HTTP /query must return per-call diagnostics (not last_query_report)."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        db = self.root / "mesh.db"
        self.mesh = MemoryMesh(Settings(db_path=db, embedding_backend="fallback"))
        self.mesh.init()

        # Project A: one versioned + one legacy (same dim) + one incompatible dim
        self.mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="cursor",
                    project="proj-a",
                    conversation_id="c",
                    role="user",
                    content="alpha-a",
                ),
                MemoryRecord(
                    provider="cursor",
                    project="proj-a",
                    conversation_id="c",
                    role="user",
                    content="legacy-a",
                ),
                MemoryRecord(
                    provider="cursor",
                    project="proj-a",
                    conversation_id="c",
                    role="user",
                    content="bad-a",
                ),
            ]
        )
        rows_a = list(self.mesh.store.list_messages("proj-a"))
        by_content = {r["content"]: int(r["id"]) for r in rows_a}
        good = serialize_embedding([1.0] + [0.0] * 127, backend="fallback", model=BACKEND_HASH_V1)
        self.mesh.store.save_embedding(by_content["alpha-a"], good)
        self.mesh.store.save_embedding(by_content["legacy-a"], json.dumps([1.0] + [0.0] * 127))
        self.mesh.store.save_embedding(by_content["bad-a"], json.dumps([0.1] * 384))

        # Project B: two malformed only
        self.mesh.store.insert_messages(
            [
                MemoryRecord(
                    provider="cursor",
                    project="proj-b",
                    conversation_id="c",
                    role="user",
                    content="bad1-b",
                ),
                MemoryRecord(
                    provider="cursor",
                    project="proj-b",
                    conversation_id="c",
                    role="user",
                    content="bad2-b",
                ),
            ]
        )
        rows_b = list(self.mesh.store.list_messages("proj-b"))
        for row in rows_b:
            self.mesh.store.save_embedding(int(row["id"]), "{not-json")

        from deepiri_memorymesh.service_api import MemoryMeshHandler, make_http_server

        class _Silent(MemoryMeshHandler):
            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        self.server = make_http_server("127.0.0.1", 0, _Silent)
        self.server.mesh = self.mesh  # type: ignore[attr-defined]
        self.server.ingest_roots = []  # type: ignore[attr-defined]
        # T37 auth is out of scope for this diagnostics test; disable it.
        self.server.auth_mode = "off"  # type: ignore[attr-defined]
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive(), "HTTP server thread did not terminate")
        self._tmpdir.cleanup()

    def _post_query(self, project: str) -> dict:
        import http.client

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = json.dumps({"project": project, "q": "alpha", "top_k": 8}).encode("utf-8")
        conn.request(
            "POST",
            "/query",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        conn.close()
        self.assertEqual(resp.status, 200, data)
        return data

    def test_concurrent_queries_keep_separate_diagnostics(self) -> None:
        barrier = threading.Barrier(2)
        results: dict[str, dict] = {}
        errors: list[BaseException] = []

        def worker(key: str, project: str) -> None:
            try:
                barrier.wait(timeout=5)
                results[key] = self._post_query(project)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with self.assertLogs("deepiri_memorymesh.sync_service", level="WARNING"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                t1 = threading.Thread(target=worker, args=("a", "proj-a"))
                t2 = threading.Thread(target=worker, args=("b", "proj-b"))
                t1.start()
                t2.start()
                t1.join(timeout=10)
                t2.join(timeout=10)
        self.assertEqual(errors, [])
        a = results["a"]
        b = results["b"]
        self.assertEqual(a["scored_versioned"], 1)
        self.assertEqual(a["legacy_compatible_embeddings"], 1)
        self.assertEqual(a["skipped_incompatible"], 1)
        self.assertEqual(a["skipped_malformed"], 0)
        self.assertIn("legacy", a.get("diagnostic", ""))
        self.assertIn("re-embed", a.get("diagnostic", "").lower())

        self.assertEqual(b["scored_versioned"], 0)
        self.assertEqual(b["legacy_compatible_embeddings"], 0)
        self.assertEqual(b["skipped_malformed"], 2)
        self.assertEqual(b["skipped_incompatible"], 0)
        self.assertIn("malformed", b.get("diagnostic", ""))
        # Responses must not swap diagnostics despite shared mesh.last_query_report.
        self.assertNotEqual(a["skipped_malformed"], b["skipped_malformed"])


if __name__ == "__main__":
    unittest.main()
