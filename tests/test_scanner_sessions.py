"""Tests for the machine-wide on-disk session lister (SDK addition)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from deepiri_memorymesh import scanner
from deepiri_memorymesh.device_paths import ProviderRoot


def _claude_line(text: str, session_id: str = "c1") -> str:
    return json.dumps(
        {
            "type": "user",
            "sessionId": session_id,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        }
    )


class ListDeviceSessionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_lists_chat_jsonl_and_filters_noise(self) -> None:
        conv = self.root / "claude" / "projects" / "-demo" / "c1.jsonl"
        conv.parent.mkdir(parents=True, exist_ok=True)
        conv.write_text(_claude_line("hello from claude") + "\n", encoding="utf-8")

        noise = self.root / "opencode" / "storage" / "session_diff" / "z.json"
        noise.parent.mkdir(parents=True, exist_ok=True)
        noise.write_text(json.dumps({"file": "a", "patch": "diff..."}), encoding="utf-8")

        sqlite = self.root / "cursor" / "globalStorage" / "state.vscdb"
        sqlite.parent.mkdir(parents=True, exist_ok=True)
        sqlite.write_bytes(b"not json")

        roots = [
            ProviderRoot(
                provider="claude",
                path=conv.parent,
                kind="json_tree",
                description="t",
                globs=["**/*.jsonl", "**/*.json"],
                exists=True,
            ),
            ProviderRoot(
                provider="opencode",
                path=noise.parent.parent,
                kind="mixed",
                description="t",
                globs=["**/*.json"],
                exists=True,
            ),
            ProviderRoot(
                provider="cursor",
                path=sqlite,
                kind="sqlite",
                description="t",
            ),
            ProviderRoot(
                provider="claude",
                path=conv,
                kind="jsonl",
                description="history",
                exists=True,
            ),
        ]
        with mock.patch.object(scanner, "discover_provider_roots", return_value=roots):
            sessions = scanner.list_device_sessions()

        self.assertEqual(len(sessions), 1, [s.path for s in sessions])
        s = sessions[0]
        self.assertEqual(s.provider, "claude")
        self.assertEqual(s.conversation_id, "c1")
        self.assertEqual(s.workspace, "demo")
        self.assertIn("hello from claude", s.preview)

    def test_dedupes_resolved_paths_and_sorts_newest_first(self) -> None:
        older = self.root / "a" / "one.jsonl"
        newer = self.root / "b" / "two.jsonl"
        older.parent.mkdir(parents=True, exist_ok=True)
        newer.parent.mkdir(parents=True, exist_ok=True)
        older.write_text(_claude_line("first", "one") + "\n", encoding="utf-8")
        newer.write_text(_claude_line("second", "two") + "\n", encoding="utf-8")

        roots = [
            ProviderRoot("claude", older.parent, "json_tree", "t", globs=["**/*.jsonl"], exists=True),
            ProviderRoot("claude", older, "jsonl", "t", exists=True),
            ProviderRoot("claude", newer.parent, "json_tree", "t", globs=["**/*.jsonl"], exists=True),
        ]
        with mock.patch.object(scanner, "discover_provider_roots", return_value=roots):
            sessions = scanner.list_device_sessions()

        self.assertEqual(len(sessions), 2)
        self.assertEqual([s.conversation_id for s in sessions], ["two", "one"])

    def test_plain_text_transcripts_are_listed(self) -> None:
        transcript = self.root / "cursor" / "projects" / "demo" / "agent-transcripts" / "cts.txt"
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text("USER: hello there\nASSISTANT: hi", encoding="utf-8")
        roots = [
            ProviderRoot(
                "cursor",
                transcript.parent.parent,
                "json_tree",
                "t",
                globs=["**/*.txt", "**/*.json", "**/*.jsonl"],
                exists=True,
            ),
        ]
        with mock.patch.object(scanner, "discover_provider_roots", return_value=roots):
            sessions = scanner.list_device_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].provider, "cursor")
        self.assertIn("hello there", sessions[0].preview)

    def test_unparseable_files_are_skipped(self) -> None:
        junk = self.root / "oc" / "x.jsonl"
        junk.parent.mkdir(parents=True, exist_ok=True)
        junk.write_text("this is not json at all\nand neither is this\n", encoding="utf-8")
        roots = [
            ProviderRoot("opencode", junk.parent, "mixed", "t", globs=["**/*.jsonl"], exists=True),
        ]
        with mock.patch.object(scanner, "discover_provider_roots", return_value=roots):
            self.assertEqual(scanner.list_device_sessions(), [])


if __name__ == "__main__":
    unittest.main()