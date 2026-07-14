"""Runtime tests for the generated OpenCode TypeScript plugin (esbuild + Node).

esbuild must never be silently downloaded via bare ``npx``. Resolution order:
local ``node_modules/.bin/esbuild``, ``esbuild`` on PATH, then
``npx --no-install esbuild`` only when a local install is already present.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from deepiri_memorymesh.integrations import install_native_integration

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_esbuild() -> list[str]:
    """Return an argv prefix that invokes esbuild without network install."""
    local_bin = _REPO_ROOT / "node_modules" / ".bin" / "esbuild"
    if local_bin.is_file():
        return [str(local_bin)]

    which = shutil.which("esbuild")
    if which:
        return [which]

    npx = shutil.which("npx")
    if npx is None:
        raise unittest.SkipTest(
            "OpenCode runtime tests skipped: Node/npx unavailable. "
            "Install Node 20+ and run `npm ci` in the repo root (pinned esbuild)."
        )

    # Only use a preinstalled package; never `npx --yes` (network install).
    probe = subprocess.run(
        [npx, "--no-install", "esbuild", "--version"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    if probe.returncode != 0:
        raise unittest.SkipTest(
            "OpenCode runtime tests skipped: esbuild not installed locally. "
            "Run `npm ci` in the repo root (see package.json). "
            f"npx --no-install stderr: {(probe.stderr or probe.stdout).strip()[:200]}"
        )
    return [npx, "--no-install", "esbuild"]


def _require_esbuild() -> str:
    if shutil.which("node") is None:
        raise unittest.SkipTest(
            "OpenCode runtime tests skipped: node binary unavailable. "
            "Install Node 20+ to run compiled plugin tests."
        )
    cmd = _resolve_esbuild()
    proc = subprocess.run(
        [*cmd, "--version"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    if proc.returncode != 0:
        raise unittest.SkipTest(
            f"OpenCode runtime tests skipped: esbuild failed: {proc.stderr}"
        )
    return proc.stdout.strip()


class OpenCodeCompiledPluginRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._esbuild_cmd = _resolve_esbuild()
        _require_esbuild()

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.home = Path(self._tmpdir.name)
        self.work = self.home / "work"
        self.work.mkdir()
        self._home_patch = mock.patch.dict(os.environ, {"HOME": str(self.home)}, clear=False)
        self._home_patch.start()
        paths = install_native_integration(
            "opencode",
            project="runtime-proj",
            service_url="http://127.0.0.1:8765",
        )
        self.plugin_ts = paths[0]
        self.bundle_js = self.work / "memorymesh.bundle.mjs"
        self._compile_plugin()

    def tearDown(self) -> None:
        self._home_patch.stop()
        self._tmpdir.cleanup()

    def _compile_plugin(self) -> None:
        proc = subprocess.run(
            [
                *self._esbuild_cmd,
                str(self.plugin_ts),
                "--bundle",
                "--format=esm",
                "--platform=neutral",
                f"--outfile={self.bundle_js}",
                "--external:@opencode-ai/plugin",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(self.bundle_js.exists())
        text = self.bundle_js.read_text(encoding="utf-8")
        self.assertNotIn(".claude", text)
        self.assertNotIn("transcript_path", text)

    def _run_harness(self, harness_body: str) -> dict:
        harness = self.work / "harness.mjs"
        header = (
            "import { pathToFileURL } from \"node:url\";\n"
            f"const mod = await import(pathToFileURL({json.dumps(str(self.bundle_js))}).href);\n"
            "const { MemoryMeshPlugin } = mod;\n"
        )
        harness.write_text(header + harness_body, encoding="utf-8")
        result_path = self.work / "result.json"
        env = {**os.environ, "HOME": str(self.home)}
        env.pop("MEMORYMESH_PROJECT", None)
        env.pop("MEMORYMESH_URL", None)
        proc = subprocess.run(
            ["node", str(harness)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(self.work),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr + "\n" + proc.stdout)
        return json.loads(result_path.read_text(encoding="utf-8"))

    def test_session_status_idle_posts_exact_payload(self) -> None:
        result = self._run_harness(
            textwrap.dedent(
                f"""\
                const calls = [];
                const fetches = [];
                const rows = [
                  {{
                    info: {{
                      id: "msg_user_1",
                      role: "user",
                      sessionID: "ses_abc",
                      time: {{ created: 1700000000000 }},
                    }},
                    parts: [
                      {{ type: "text", text: "hello" }},
                      {{ type: "tool", tool: "bash" }},
                      {{ type: "text", text: "world" }},
                    ],
                  }},
                  {{
                    info: {{
                      id: "msg_asst_1",
                      role: "assistant",
                      sessionID: "ses_abc",
                      time: {{ created: 1700000001000 }},
                    }},
                    parts: [
                      {{ type: "reasoning", text: "think" }},
                      {{ type: "text", text: "reply" }},
                    ],
                  }},
                  {{
                    info: {{ id: "msg_empty", role: "assistant" }},
                    parts: [{{ type: "tool", tool: "read" }}],
                  }},
                ];
                const client = {{
                  session: {{
                    messages: async ({{ path }}) => {{
                      calls.push(path);
                      return rows;
                    }},
                  }},
                }};
                globalThis.fetch = async (url, init) => {{
                  fetches.push({{ url, body: init.body, headers: init.headers }});
                  return {{ ok: true, status: 200 }};
                }};
                const plugin = await MemoryMeshPlugin({{ client }});
                await plugin.event({{
                  event: {{
                    type: "session.status",
                    properties: {{ sessionID: "ses_abc", status: {{ type: "idle" }} }},
                  }},
                }});
                const payload = JSON.parse(fetches[0].body);
                await import("node:fs").then((fs) =>
                  fs.writeFileSync(
                    {json.dumps(str(self.work / "result.json"))},
                    JSON.stringify({{
                      calls,
                      fetchCount: fetches.length,
                      url: fetches[0].url,
                      payload,
                    }}),
                  ),
                );
                """
            )
        )
        self.assertEqual(result["calls"], [{"id": "ses_abc"}])
        self.assertEqual(result["fetchCount"], 1)
        self.assertEqual(result["url"], "http://127.0.0.1:8765/ingest")
        payload = result["payload"]
        self.assertEqual(payload["provider"], "opencode")
        self.assertEqual(payload["project"], "runtime-proj")
        self.assertNotIn("file_path", payload)
        messages = payload["conversation"]["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "hello\nworld")
        self.assertEqual(messages[0]["metadata"]["message_id"], "msg_user_1")
        self.assertEqual(messages[0]["metadata"]["session_id"], "ses_abc")
        self.assertEqual(messages[0]["timestamp"], "2023-11-14T22:13:20.000Z")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "reply")

    def test_data_wrapper_shape_and_dedupe_with_legacy_idle(self) -> None:
        result = self._run_harness(
            textwrap.dedent(
                f"""\
                let messageCalls = 0;
                const fetches = [];
                const client = {{
                  session: {{
                    messages: async ({{ path }}) => {{
                      messageCalls += 1;
                      return {{
                        data: [
                          {{
                            info: {{ id: "m1", role: "user", time: {{ created: 1700000000000 }} }},
                            parts: [{{ type: "text", text: "hi" }}],
                          }},
                        ],
                      }};
                    }},
                  }},
                }};
                globalThis.fetch = async (url, init) => {{
                  fetches.push(init.body);
                  return {{ ok: true, status: 200 }};
                }};
                const plugin = await MemoryMeshPlugin({{ client }});
                await plugin.event({{
                  event: {{
                    type: "session.status",
                    properties: {{ sessionID: "ses_1", status: {{ type: "idle" }} }},
                  }},
                }});
                await plugin.event({{
                  event: {{ type: "session.idle", properties: {{ sessionID: "ses_1" }} }},
                }});
                await import("node:fs").then((fs) =>
                  fs.writeFileSync(
                    {json.dumps(str(self.work / "result.json"))},
                    JSON.stringify({{ messageCalls, fetchCount: fetches.length }}),
                  ),
                );
                """
            )
        )
        self.assertEqual(result["messageCalls"], 1)
        self.assertEqual(result["fetchCount"], 1)

    def test_non_idle_resets_and_later_idle_ingests_again(self) -> None:
        result = self._run_harness(
            textwrap.dedent(
                f"""\
                let messageCalls = 0;
                const fetches = [];
                const client = {{
                  session: {{
                    messages: async () => {{
                      messageCalls += 1;
                      return [
                        {{
                          info: {{ id: "m1", role: "user" }},
                          parts: [{{ type: "text", text: "x" }}],
                        }},
                      ];
                    }},
                  }},
                }};
                globalThis.fetch = async (_url, init) => {{
                  fetches.push(init.body);
                  return {{ ok: true, status: 200 }};
                }};
                const plugin = await MemoryMeshPlugin({{ client }});
                await plugin.event({{
                  event: {{
                    type: "session.status",
                    properties: {{ sessionID: "ses_1", status: {{ type: "idle" }} }},
                  }},
                }});
                await plugin.event({{
                  event: {{
                    type: "session.status",
                    properties: {{ sessionID: "ses_1", status: {{ type: "busy" }} }},
                  }},
                }});
                await plugin.event({{
                  event: {{
                    type: "session.status",
                    properties: {{ sessionID: "ses_1", status: {{ type: "idle" }} }},
                  }},
                }});
                await import("node:fs").then((fs) =>
                  fs.writeFileSync(
                    {json.dumps(str(self.work / "result.json"))},
                    JSON.stringify({{ messageCalls, fetchCount: fetches.length }}),
                  ),
                );
                """
            )
        )
        self.assertEqual(result["messageCalls"], 2)
        self.assertEqual(result["fetchCount"], 2)

    def test_failed_retrieval_does_not_fetch_and_can_retry(self) -> None:
        result = self._run_harness(
            textwrap.dedent(
                f"""\
                let messageCalls = 0;
                const fetches = [];
                const client = {{
                  session: {{
                    messages: async () => {{
                      messageCalls += 1;
                      if (messageCalls === 1) throw new Error("boom");
                      return [
                        {{
                          info: {{ id: "m1", role: "user" }},
                          parts: [{{ type: "text", text: "ok" }}],
                        }},
                      ];
                    }},
                  }},
                }};
                globalThis.fetch = async (_url, init) => {{
                  fetches.push(init.body);
                  return {{ ok: true, status: 200 }};
                }};
                const plugin = await MemoryMeshPlugin({{ client }});
                await plugin.event({{
                  event: {{ type: "session.idle", properties: {{ sessionID: "ses_retry" }} }},
                }});
                await plugin.event({{
                  event: {{ type: "session.idle", properties: {{ sessionID: "ses_retry" }} }},
                }});
                await import("node:fs").then((fs) =>
                  fs.writeFileSync(
                    {json.dumps(str(self.work / "result.json"))},
                    JSON.stringify({{ messageCalls, fetchCount: fetches.length }}),
                  ),
                );
                """
            )
        )
        self.assertEqual(result["messageCalls"], 2)
        self.assertEqual(result["fetchCount"], 1)

    def test_failed_http_does_not_crash_and_can_retry(self) -> None:
        result = self._run_harness(
            textwrap.dedent(
                f"""\
                let fetches = 0;
                const client = {{
                  session: {{
                    messages: async () => [
                      {{
                        info: {{ id: "m1", role: "user" }},
                        parts: [{{ type: "text", text: "ok" }}],
                      }},
                    ],
                  }},
                }};
                globalThis.fetch = async () => {{
                  fetches += 1;
                  if (fetches === 1) return {{ ok: false, status: 500 }};
                  return {{ ok: true, status: 200 }};
                }};
                const plugin = await MemoryMeshPlugin({{ client }});
                await plugin.event({{
                  event: {{ type: "session.idle", properties: {{ sessionID: "ses_http" }} }},
                }});
                await plugin.event({{
                  event: {{ type: "session.idle", properties: {{ sessionID: "ses_http" }} }},
                }});
                await import("node:fs").then((fs) =>
                  fs.writeFileSync(
                    {json.dumps(str(self.work / "result.json"))},
                    JSON.stringify({{ fetches }}),
                  ),
                );
                """
            )
        )
        self.assertEqual(result["fetches"], 2)

    def test_missing_session_id_fails_closed(self) -> None:
        result = self._run_harness(
            textwrap.dedent(
                f"""\
                let messageCalls = 0;
                let fetches = 0;
                const client = {{
                  session: {{
                    messages: async () => {{
                      messageCalls += 1;
                      return [];
                    }},
                  }},
                }};
                globalThis.fetch = async () => {{
                  fetches += 1;
                  return {{ ok: true, status: 200 }};
                }};
                const plugin = await MemoryMeshPlugin({{ client }});
                await plugin.event({{
                  event: {{ type: "session.idle", properties: {{}} }},
                }});
                await plugin.event({{
                  event: {{
                    type: "session.status",
                    properties: {{ status: {{ type: "idle" }} }},
                  }},
                }});
                await import("node:fs").then((fs) =>
                  fs.writeFileSync(
                    {json.dumps(str(self.work / "result.json"))},
                    JSON.stringify({{ messageCalls, fetches }}),
                  ),
                );
                """
            )
        )
        self.assertEqual(result["messageCalls"], 0)
        self.assertEqual(result["fetches"], 0)


if __name__ == "__main__":
    unittest.main()
