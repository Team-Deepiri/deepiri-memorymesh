"""Focused regression tests for T04/T05/T07/T10/T11/T17/T21 corrections."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from deepiri_memorymesh.cli import app
from deepiri_memorymesh.config import Settings
from deepiri_memorymesh.integrations import (
    build_ingest_payload,
    install_bridge_script,
    install_hook_script,
    install_native_integration,
    write_hook_snippets,
    write_integration_template,
)
from deepiri_memorymesh.models import AgentState, CompressedRecord, MemoryRecord
from deepiri_memorymesh.providers.cursor import parse_cursor_file
from deepiri_memorymesh.storage import MemoryStore
from deepiri_memorymesh.sync_service import MemoryMesh, SyncFileFailure


UNSAFE_JSON_CURL_PATTERN = '-d "{\\"provider\\"'


def _adversarial_project(marker: Path) -> str:
    return (
        f"proj $(touch {marker}) `touch {marker}` ; "
        f"\"dq\" 'sq' \\back unicodé-路径\nnewline -leading"
    )


def _adversarial_url(marker: Path) -> str:
    return f"http://127.0.0.1:8765/$(touch {marker})"


class ForeignKeyTests(unittest.TestCase):
    """T17."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmpdir.name) / "mesh.db")
        self.store.init()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_pragma_foreign_keys_enabled(self) -> None:
        with self.store.connection() as conn:
            value = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        self.assertEqual(value, 1)

    def test_each_new_connection_enables_and_closes(self) -> None:
        for _ in range(3):
            with self.store.connection() as conn:
                self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)

    def test_embedding_for_missing_message_fails(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save_embedding(999999, "[0.1,0.2]")

    def test_embedding_for_valid_message_works(self) -> None:
        inserted = self.store.insert_messages(
            [
                MemoryRecord(
                    provider="cursor",
                    project="p",
                    conversation_id="c1",
                    role="user",
                    content="hello",
                )
            ]
        )
        self.assertEqual(inserted, 1)
        msg_id = self.store.list_messages("p")[0]["id"]
        self.store.save_embedding(int(msg_id), "[1.0,0.0]")
        self.assertEqual(self.store.project_stats("p")["embeddings"], 1)

    def test_normal_workflows_still_function(self) -> None:
        self.store.insert_messages(
            [
                MemoryRecord(
                    provider="claude",
                    project="p",
                    conversation_id="c1",
                    role="user",
                    content="alpha",
                )
            ]
        )
        self.store.upsert_summary(
            CompressedRecord(project="p", conversation_id="c1", summary="s", method="m")
        )
        self.store.set_agent_state(AgentState(project="p", agent="a", key="k", value="v"))
        self.assertEqual(self.store.get_agent_state("p", "a", "k"), "v")
        self.assertEqual(len(self.store.list_summaries("p")), 1)

    def test_connection_context_closes(self) -> None:
        with self.store.connection() as conn:
            raw = conn
        with self.assertRaises(sqlite3.ProgrammingError):
            raw.execute("SELECT 1")


class UpsertSummaryTests(unittest.TestCase):
    """T07."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmpdir.name) / "mesh.db")
        self.store.init()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _count(self, project: str, conversation_id: str) -> int:
        with self.store.connection() as conn:
            return int(
                conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM memory_summaries
                    WHERE project = ? AND conversation_id = ?
                    """,
                    (project, conversation_id),
                ).fetchone()["c"]
            )

    def test_first_insert_then_update_same_row(self) -> None:
        self.store.upsert_summary(
            CompressedRecord(
                project="p",
                conversation_id="c1",
                summary="first",
                method="m1",
                created_at="2020-01-01T00:00:00+00:00",
            )
        )
        self.assertEqual(self._count("p", "c1"), 1)
        self.store.upsert_summary(
            CompressedRecord(
                project="p",
                conversation_id="c1",
                summary="second",
                method="m2",
                created_at="2020-01-02T00:00:00+00:00",
            )
        )
        self.assertEqual(self._count("p", "c1"), 1)
        row = self.store.list_summaries("p")[0]
        self.assertEqual(row["summary"], "second")
        self.assertEqual(row["method"], "m2")
        self.assertEqual(row["created_at"], "2020-01-02T00:00:00+00:00")

    def test_different_conversations_and_projects_separate(self) -> None:
        self.store.upsert_summary(
            CompressedRecord(project="p1", conversation_id="c1", summary="a", method="m")
        )
        self.store.upsert_summary(
            CompressedRecord(project="p1", conversation_id="c2", summary="b", method="m")
        )
        self.store.upsert_summary(
            CompressedRecord(project="p2", conversation_id="c1", summary="c", method="m")
        )
        self.assertEqual(self._count("p1", "c1"), 1)
        self.assertEqual(self._count("p1", "c2"), 1)
        self.assertEqual(self._count("p2", "c1"), 1)

    def test_preexisting_duplicates_all_updated_deterministically(self) -> None:
        with self.store.connection() as conn:
            conn.executemany(
                """
                INSERT INTO memory_summaries
                (project, conversation_id, summary, method, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    ("p", "c1", "old1", "m", "t1"),
                    ("p", "c1", "old2", "m", "t2"),
                ],
            )
            conn.commit()
        self.assertEqual(self._count("p", "c1"), 2)
        self.store.upsert_summary(
            CompressedRecord(
                project="p",
                conversation_id="c1",
                summary="canonical",
                method="updated",
                created_at="t3",
            )
        )
        self.assertEqual(self._count("p", "c1"), 2)
        with self.store.connection() as conn:
            rows = conn.execute(
                """
                SELECT summary, method, created_at FROM memory_summaries
                WHERE project = ? AND conversation_id = ?
                """,
                ("p", "c1"),
            ).fetchall()
        self.assertEqual(
            {(r["summary"], r["method"], r["created_at"]) for r in rows},
            {("canonical", "updated", "t3")},
        )

    def test_concurrent_upserts_do_not_duplicate(self) -> None:
        for round_idx in range(5):
            conv = f"c-concurrent-{round_idx}"
            errors: list[BaseException] = []
            barrier = threading.Barrier(4)
            summaries = [f"s{i}" for i in range(4)]

            def worker(summary: str) -> None:
                store = MemoryStore(self.store.db_path)
                try:
                    barrier.wait(timeout=5)
                    store.upsert_summary(
                        CompressedRecord(
                            project="p",
                            conversation_id=conv,
                            summary=summary,
                            method="m",
                            created_at="2020-01-01T00:00:00+00:00",
                        )
                    )
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(s,)) for s in summaries]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)
                self.assertFalse(t.is_alive(), f"thread still alive for {conv}")
            self.assertEqual(errors, [])
            self.assertEqual(self._count("p", conv), 1)
            row = [
                r for r in self.store.list_summaries("p") if r["conversation_id"] == conv
            ][0]
            self.assertIn(row["summary"], set(summaries))
            self.assertEqual(row["method"], "m")


class CursorJsonlResilienceTests(unittest.TestCase):
    """T21."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_skips_malformed_and_blank_preserves_valid(self) -> None:
        path = self.root / "chat.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"role": "user", "content": "one"}),
                    "{not-json",
                    "",
                    json.dumps({"role": "assistant", "content": "two"}),
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertLogs("deepiri_memorymesh.providers.cursor", level="WARNING") as cm:
            records = parse_cursor_file("cursor", "proj", path)
        self.assertEqual([r.content for r in records], ["one", "two"])
        joined = "\n".join(cm.output)
        self.assertIn("line 2", joined)
        self.assertNotIn("{not-json", joined)

    def test_valid_jsonl_unchanged(self) -> None:
        path = self.root / "ok.jsonl"
        path.write_text(json.dumps({"role": "user", "content": "only"}) + "\n", encoding="utf-8")
        records = parse_cursor_file("cursor", "proj", path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].content, "only")


class SyncFailureReportingTests(unittest.TestCase):
    """T10."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        self.db = self.root / "db.sqlite"
        self.src = self.root / "src"
        self.src.mkdir()
        settings = Settings(db_path=self.db, embedding_backend="fallback")
        self.mesh = MemoryMesh(settings)
        self.mesh.init()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_partial_sync_reports_failures_and_keeps_compat(self) -> None:
        good = self.src / "good.jsonl"
        bad = self.src / "nested" / "bad.json"
        bad.parent.mkdir()
        good.write_text(json.dumps({"role": "user", "content": "ok-msg"}) + "\n", encoding="utf-8")
        bad.write_text("{broken\n", encoding="utf-8")

        sink: list[SyncFileFailure] = []
        processed, inserted = self.mesh.sync_directory(
            "cursor", "proj", self.src, recursive=True, error_sink=sink
        )
        self.assertEqual(processed, 1)
        self.assertGreaterEqual(inserted, 1)
        self.assertEqual(len(sink), 1)
        self.assertEqual(sink[0].path, bad)
        self.assertEqual(sink[0].error_type, "JSONDecodeError")

        report = self.mesh.sync_directory_report("cursor", "proj", self.src, recursive=True)
        self.assertEqual(report.attempted, 2)
        self.assertEqual(report.processed, 1)
        self.assertEqual(report.failed, 1)

    def test_on_error_callback_exception_does_not_abort(self) -> None:
        good = self.src / "z-good.jsonl"
        bad = self.src / "a-bad.json"
        good.write_text(json.dumps({"role": "user", "content": "later-ok"}) + "\n", encoding="utf-8")
        bad.write_text("{broken\n", encoding="utf-8")

        def boom(_failure: SyncFileFailure) -> None:
            raise RuntimeError("callback exploded")

        with self.assertLogs("deepiri_memorymesh.sync_service", level="WARNING") as cm:
            report = self.mesh.sync_directory_report(
                "cursor", "proj", self.src, recursive=True, on_error=boom
            )
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.processed, 1)
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].error_type, "JSONDecodeError")
        joined = "\n".join(cm.output)
        self.assertIn("on_error callback failed", joined)
        self.assertIn("callback exploded", joined)
        self.assertIn("a-bad.json", joined)
        self.assertNotIn("Traceback", joined)
        self.assertGreaterEqual(report.inserted, 1)

    def test_cli_sync_shows_failure_without_traceback(self) -> None:
        good = self.src / "chat-good.jsonl"
        bad = self.src / "chat-bad.json"
        good.write_text(json.dumps({"role": "user", "content": "ok-msg"}) + "\n", encoding="utf-8")
        bad.write_text("{broken\n", encoding="utf-8")
        runner = CliRunner()
        with mock.patch("deepiri_memorymesh.cli._mesh", return_value=self.mesh):
            result = runner.invoke(
                app,
                [
                    "sync",
                    "--provider",
                    "cursor",
                    "--project",
                    "proj",
                    "--source-dir",
                    str(self.src),
                ],
            )
        combined = (result.stdout or "") + (result.stderr or "")
        self.assertEqual(result.exit_code, 0, combined)
        self.assertIn("FAILED", combined)
        self.assertIn("chat-bad.json", combined)
        self.assertIn("failed 1", combined)
        self.assertNotIn("Traceback", combined)


class ShellAndJsonSecurityTests(unittest.TestCase):
    """T11 + generated JSON safety — scripts must execute with embedded defaults."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.home = Path(self._tmpdir.name)
        self.bin_dir = self.home / ".local" / "bin"
        self.bin_dir.mkdir(parents=True)
        self.marker = self.home / "PWNED"
        self.capture_body = self.home / "captured.json"
        self.capture_url = self.home / "captured.url"
        self._home_patch = mock.patch.dict(os.environ, {"HOME": str(self.home)}, clear=False)
        self._home_patch.start()

    def tearDown(self) -> None:
        self._home_patch.stop()
        self._tmpdir.cleanup()

    def _install_fake_curl(self) -> None:
        curl = self.bin_dir / "curl"
        curl.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f'BODY_OUT="{self.capture_body}"\n'
            f'URL_OUT="{self.capture_url}"\n'
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
            "exit 0\n",
            encoding="utf-8",
        )
        curl.chmod(0o755)

    def _clean_env(self) -> dict[str, str]:
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"MEMORYMESH_PROJECT", "MEMORYMESH_URL"}
        }
        env["HOME"] = str(self.home)
        env["PATH"] = f"{self.bin_dir}:{env.get('PATH', '')}"
        return env

    def test_build_ingest_payload_round_trip(self) -> None:
        for value in [
            'path with spaces/"quotes"/\'sq\'',
            "back\\slash",
            "unicodé-路径",
            "line\nbreak",
            "$(echo injected)",
            "semi;colon",
            "-leading-hyphen.json",
        ]:
            payload = build_ingest_payload("cursor", 'proj "x"', value)
            parsed = json.loads(payload)
            self.assertEqual(parsed["file_path"], value)

    def test_bridge_executes_with_malicious_embedded_defaults(self) -> None:
        self._install_fake_curl()
        project = _adversarial_project(self.marker)
        service_url = _adversarial_url(self.marker)
        bridge = install_bridge_script("cursor", project, service_url)
        body = bridge.read_text(encoding="utf-8")
        self.assertNotIn(UNSAFE_JSON_CURL_PATTERN, body)
        self.assertIn("DEFAULT_PROJECT=", body)
        self.assertIn("--data-binary @-", body)

        target = self.home / "-leading file with spaces.json"
        target.write_text("{}", encoding="utf-8")
        proc = subprocess.run(
            ["bash", str(bridge), str(target)],
            check=False,
            capture_output=True,
            text=True,
            env=self._clean_env(),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
        self.assertFalse(self.marker.exists(), "command injection created marker file")
        payload = json.loads(self.capture_body.read_text(encoding="utf-8"))
        self.assertEqual(payload["provider"], "cursor")
        self.assertEqual(payload["project"], project)
        self.assertEqual(payload["file_path"], str(target))
        self.assertEqual(self.capture_url.read_text(encoding="utf-8"), f"{service_url}/ingest")

    def test_hook_executes_with_malicious_embedded_defaults(self) -> None:
        self._install_fake_curl()
        project = _adversarial_project(self.marker)
        service_url = _adversarial_url(self.marker)
        hook = install_hook_script("gemini", project, service_url)
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
        self.assertFalse(self.marker.exists())
        payload = json.loads(self.capture_body.read_text(encoding="utf-8"))
        self.assertEqual(payload["project"], project)
        self.assertEqual(payload["file_path"], str(export))
        self.assertEqual(self.capture_url.read_text(encoding="utf-8"), f"{service_url}/ingest")

    def test_aider_wrapper_injection_does_not_execute(self) -> None:
        project = _adversarial_project(self.marker)
        paths = install_native_integration("aider", project)
        wrapper = next(p for p in paths if p.name == "aider-memorymesh")
        # Fake aider + memorymesh on PATH.
        aider = self.bin_dir / "aider"
        aider.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        aider.chmod(0o755)
        mm = self.bin_dir / "memorymesh"
        mm.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        mm.chmod(0o755)
        proc = subprocess.run(
            ["bash", str(wrapper)],
            check=False,
            capture_output=True,
            text=True,
            env=self._clean_env(),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(self.marker.exists())
        self.assertNotIn(".claude", wrapper.read_text(encoding="utf-8"))

    def test_hooks_never_reference_claude_except_claude(self) -> None:
        for key in ("cursor", "gemini", "continue", "jsonl"):
            text = install_hook_script(key, "proj").read_text(encoding="utf-8")
            self.assertNotIn(".claude", text, msg=key)
            self.assertNotIn(UNSAFE_JSON_CURL_PATTERN, text)
        claude = install_hook_script("claude", "proj").read_text(encoding="utf-8")
        self.assertIn(".claude/history.jsonl", claude)

    def test_generated_json_files_round_trip(self) -> None:
        project = 'p"q\\r\nunicodé $(nope)'
        template = write_integration_template("cursor", project)
        payload = json.loads(template.read_text(encoding="utf-8"))
        self.assertEqual(payload["project"], project)

        oc_template = write_integration_template("opencode", project)
        oc_payload = json.loads(oc_template.read_text(encoding="utf-8"))
        self.assertEqual(oc_payload["project"], project)
        self.assertEqual(oc_payload["integration"], "native-plugin")

        out_dir = self.home / "snippets"
        files = write_hook_snippets(project, out_dir)
        names = {p.name for p in files}
        self.assertIn("opencode.plugin.json", names)
        self.assertNotIn("opencode.hook.json", names)
        for path in files:
            if path.suffix == ".json":
                parsed = json.loads(path.read_text(encoding="utf-8"))
                if "project" in parsed:
                    self.assertEqual(parsed["project"], project)

    def test_opencode_native_installs_plugin_only_under_plugins(self) -> None:
        paths = install_native_integration("opencode", "proj", "http://127.0.0.1:8765")
        plugin = self.home / ".config" / "opencode" / "plugins" / "memorymesh.ts"
        push = self.home / ".local" / "bin" / "memorymesh-push-opencode"
        self.assertIn(plugin, paths)
        self.assertIn(push, paths)
        self.assertEqual(set(paths), {plugin, push})
        text = plugin.read_text(encoding="utf-8")
        self.assertNotIn(".claude", text)
        self.assertNotIn("transcript_path", text)
        self.assertIn("session.status", text)
        self.assertIn("session.idle", text)
        self.assertIn("client.session.messages", text)
        self.assertFalse((self.home / ".local" / "bin" / "memorymesh-bridge-opencode").exists())
        self.assertFalse((self.home / ".local" / "bin" / "memorymesh-hook-opencode").exists())


if __name__ == "__main__":
    unittest.main()
