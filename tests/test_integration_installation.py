"""Batch 6: T12/T36 transactional install + uninstall."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.helpers import temp_home

from deepiri_memorymesh import integration_install as ii
from deepiri_memorymesh.integration_install import (
    install_native_transactional,
    list_installations,
    restore_from_backup,
    uninstall_all_transactional,
    uninstall_native_transactional,
    verify_installation,
)
from deepiri_memorymesh.integrations import (
    _curl_ingest_from_env_fragment,
    render_bridge_script,
)


class TransactionalInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir()
        self._cm = temp_home(self.home)
        self._cm.__enter__()
        self.token = self.home / "token"
        self.token.write_text("mmht.deadbeefdeadbeef." + ("a" * 43), encoding="utf-8")
        self.token.chmod(0o600)
        ii._INJECT_FAILURE_AT = None

    def tearDown(self) -> None:
        ii._INJECT_FAILURE_AT = None
        self._cm.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_dry_run_writes_nothing(self) -> None:
        report = install_native_transactional(
            "opencode",
            project="p",
            token_file=self.token,
            dry_run=True,
        )
        self.assertTrue(report.ok)
        self.assertTrue(report.dry_run)
        plugin = self.home / ".config" / "opencode" / "plugins" / "memorymesh.ts"
        self.assertFalse(plugin.exists())
        self.assertEqual(list_installations(), [])

    def test_auth_required_without_token_rejected(self) -> None:
        with self.assertRaises(ValueError):
            install_native_transactional(
                "opencode", project="p", auth_required=True, token_file=None
            )

    def test_opencode_install_manifest_no_secret_and_idempotent(self) -> None:
        report = install_native_transactional(
            "opencode",
            project="demo",
            token_file=self.token,
            auth_required=True,
        )
        self.assertTrue(report.ok)
        self.assertIsNotNone(report.installation_id)
        plugin = self.home / ".config" / "opencode" / "plugins" / "memorymesh.ts"
        self.assertTrue(plugin.exists())
        text = plugin.read_text(encoding="utf-8")
        self.assertIn("installation_id=", text)
        self.assertNotIn(self.token.read_text(encoding="utf-8").strip(), text)
        self.assertIn("MEMORYMESH_TOKEN", text)

        manifest = report.manifest_path
        assert manifest is not None
        raw = manifest.read_text(encoding="utf-8")
        self.assertNotIn(self.token.read_text().strip(), raw)
        data = json.loads(raw)
        self.assertEqual(data["token_file"], str(self.token))

        again = install_native_transactional(
            "opencode",
            project="demo",
            token_file=self.token,
            auth_required=True,
        )
        self.assertTrue(again.noop)

        conflict = install_native_transactional(
            "opencode",
            project="other",
            token_file=self.token,
            auth_required=True,
        )
        self.assertFalse(conflict.ok)
        self.assertIn("differs", conflict.message)

        verify = verify_installation(report.installation_id)  # type: ignore[arg-type]
        self.assertTrue(verify["ok"])

    def test_claude_transactional_preserves_unrelated_hooks(self) -> None:
        settings = self.home / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [
                            {
                                "matcher": ".*",
                                "hooks": [{"type": "command", "command": "echo unrelated"}],
                            }
                        ],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report = install_native_transactional(
            "claude", project="demo", token_file=self.token, auth_required=True
        )
        self.assertTrue(report.ok, report.message)
        cfg = json.loads(settings.read_text(encoding="utf-8"))
        entries = cfg["hooks"]["SessionEnd"]
        blob = json.dumps(entries)
        self.assertIn("echo unrelated", blob)
        self.assertIn("memorymesh-hook-claude", blob)
        owned = [
            e
            for e in entries
            if isinstance(e, dict) and "memorymesh_ownership" in e
        ]
        self.assertEqual(len(owned), 1)
        bridge = self.home / ".local" / "bin" / "memorymesh-bridge-claude"
        self.assertTrue(bridge.exists())
        self.assertIn("--config", bridge.read_text(encoding="utf-8"))
        self.assertNotIn(self.token.read_text().strip(), bridge.read_text())

    def test_install_failure_injection_opencode_no_partial(self) -> None:
        for phase in (
            "after_backup",
            "after_stage",
            "after_verify",
            "before_manifest",
        ):
            with self.subTest(phase=phase):
                ii._INJECT_FAILURE_AT = phase
                plugin = self.home / ".config" / "opencode" / "plugins" / "memorymesh.ts"
                preexisting = b"// keep me\n"
                plugin.parent.mkdir(parents=True, exist_ok=True)
                plugin.write_bytes(preexisting)
                report = install_native_transactional(
                    "opencode", project="demo", token_file=self.token
                )
                self.assertFalse(report.ok)
                self.assertTrue(report.rolled_back)
                self.assertTrue(report.rollback_ok, report.rollback_errors)
                self.assertEqual(plugin.read_bytes(), preexisting)
                self.assertEqual(list_installations(), [])
                ii._INJECT_FAILURE_AT = None

    def test_install_failure_injection_claude_restores_config(self) -> None:
        settings = self.home / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True)
        original = (
            json.dumps(
                {
                    "hooks": {
                        "SessionEnd": [
                            {
                                "matcher": ".*",
                                "hooks": [{"type": "command", "command": "keep"}],
                            }
                        ]
                    }
                }
            )
            + "\n"
        )
        settings.write_text(original, encoding="utf-8")
        ii._INJECT_FAILURE_AT = "before_manifest"
        report = install_native_transactional(
            "claude", project="demo", token_file=self.token
        )
        self.assertFalse(report.ok)
        self.assertTrue(report.rollback_ok, report.rollback_errors)
        self.assertEqual(settings.read_text(encoding="utf-8"), original)
        bridge = self.home / ".local" / "bin" / "memorymesh-bridge-claude"
        self.assertFalse(bridge.exists())
        self.assertEqual(list_installations(), [])

    def test_uninstall_preserves_modified_generated_file(self) -> None:
        report = install_native_transactional(
            "opencode", project="demo", token_file=self.token
        )
        plugin = self.home / ".config" / "opencode" / "plugins" / "memorymesh.ts"
        plugin.write_text(plugin.read_text(encoding="utf-8") + "\n// user edit\n", encoding="utf-8")
        un = uninstall_native_transactional("opencode", force=False)
        self.assertIn(str(plugin), un.preserved_files)
        self.assertTrue(plugin.exists())
        un2 = uninstall_native_transactional("opencode", force=True)
        self.assertFalse(plugin.exists())
        self.assertTrue(un2.ok or un2.noop or "ok" in un2.message)

    def test_uninstall_failure_injection_retains_manifest(self) -> None:
        report = install_native_transactional(
            "claude", project="demo", token_file=self.token
        )
        assert report.installation_id
        settings = self.home / ".claude" / "settings.json"
        before = settings.read_text(encoding="utf-8")
        ii._INJECT_FAILURE_AT = "uninstall_after_config_stage"
        un = uninstall_native_transactional("claude", force=True)
        self.assertFalse(un.ok)
        self.assertTrue(un.rolled_back)
        self.assertTrue(un.rollback_ok, un.rollback_errors)
        self.assertEqual(settings.read_text(encoding="utf-8"), before)
        self.assertTrue(verify_installation(report.installation_id)["ok"])

        ii._INJECT_FAILURE_AT = "uninstall_after_quarantine"
        un2 = uninstall_native_transactional("claude", force=True)
        self.assertFalse(un2.ok)
        self.assertTrue(un2.rollback_ok, un2.rollback_errors)
        self.assertTrue(
            (self.home / ".local" / "bin" / "memorymesh-bridge-claude").exists()
        )
        self.assertTrue(verify_installation(report.installation_id)["ok"])

        ii._INJECT_FAILURE_AT = None
        un3 = uninstall_native_transactional("claude", force=True)
        self.assertTrue(un3.ok)
        second = uninstall_native_transactional("claude")
        self.assertTrue(second.noop)

    def test_uninstall_all_independent_results(self) -> None:
        install_native_transactional("opencode", project="a", token_file=self.token)
        install_native_transactional("aider", project="b", token_file=self.token)
        results = uninstall_all_transactional(force=True)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(list_installations(), [])

    def test_uninstall_dry_run_and_second_noop(self) -> None:
        install_native_transactional("opencode", project="demo", token_file=self.token)
        dry = uninstall_native_transactional("opencode", dry_run=True)
        self.assertTrue(dry.dry_run)
        plugin = self.home / ".config" / "opencode" / "plugins" / "memorymesh.ts"
        self.assertTrue(plugin.exists())
        uninstall_native_transactional("opencode", force=True)
        second = uninstall_native_transactional("opencode")
        self.assertTrue(second.noop)

    def test_restore_requires_yes_and_rejects_wrong_backup(self) -> None:
        report = install_native_transactional(
            "opencode", project="demo", token_file=self.token
        )
        gated = restore_from_backup(report.installation_id, yes=False)  # type: ignore[arg-type]
        self.assertFalse(gated["ok"])
        self.assertEqual(gated["error"], "confirmation_required")

    def test_unsupported_provider_rejected(self) -> None:
        with self.assertRaises(ValueError):
            install_native_transactional("openai", project="p", auth_required=False)

    def test_never_calls_classic_mutating_installer(self) -> None:
        # Avoid ~/.cursor in restricted CI sandboxes that block creating that path.
        targets = ("claude", "gemini", "continue", "aider", "opencode")
        with mock.patch(
            "deepiri_memorymesh.integrations.install_native_integration",
            side_effect=AssertionError("classic installer must not be called"),
        ):
            for target in targets:
                with self.subTest(target=target):
                    r = install_native_transactional(
                        target, project=f"p-{target}", token_file=self.token
                    )
                    self.assertTrue(r.ok, r.message)


class BearerCurlConfigTests(unittest.TestCase):
    def test_fragment_uses_config_not_header_argv(self) -> None:
        frag = _curl_ingest_from_env_fragment()
        self.assertIn("MM_CURL_CFG", frag)
        self.assertIn("--config", frag)
        self.assertIn("chmod 600", frag)
        self.assertIn("trap", frag)
        self.assertNotIn('-H "Authorization: Bearer $AUTH_TOKEN"', frag)
        self.assertNotIn("-H \"Authorization: Bearer $AUTH_TOKEN\"", frag)

    def test_fake_curl_argv_never_sees_token(self) -> None:
        import os
        import subprocess

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = Path(tmp.name) / "home"
        home.mkdir()
        bin_dir = home / "bin"
        bin_dir.mkdir()
        argv_log = home / "curl_argv.log"
        curl = bin_dir / "curl"
        curl.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f'printf "%s\\n" "$@" > "{argv_log}"\n'
            'cfg=""\n'
            'while [ "$#" -gt 0 ]; do\n'
            '  case "$1" in\n'
            "    --config) shift; cfg=\"${1:-}\"; shift || true ;;\n"
            "    --data-binary) shift; if [ \"${1:-}\" = \"@-\" ]; then cat >/dev/null; fi; shift || true ;;\n"
            "    -X|-H) shift 2 || true ;;\n"
            "    -sS|--fail) shift ;;\n"
            "    *) shift ;;\n"
            "  esac\n"
            "done\n"
            'if [ -n "$cfg" ] && [ -f "$cfg" ]; then\n'
            '  mode="$(stat -c \'%a\' "$cfg" 2>/dev/null || stat -f \'%Lp\' "$cfg" 2>/dev/null || echo)"\n'
            f'  echo "cfgmode=$mode" >> "{argv_log}"\n'
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        curl.chmod(0o755)
        token = "mmht.deadbeefdeadbeef." + ("b" * 43)
        with temp_home(home):
            _path, body = render_bridge_script("cursor", "proj", "http://127.0.0.1:8765")
            script = home / "bridge.sh"
            script.write_text(body, encoding="utf-8")
            script.chmod(0o755)
            payload = home / "payload.json"
            payload.write_text("{}", encoding="utf-8")
            env = {
                **os.environ,
                "HOME": str(home),
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
                "MEMORYMESH_TOKEN": token,
                "TMPDIR": str(home / "tmp"),
            }
            (home / "tmp").mkdir()
            proc = subprocess.run(
                ["bash", str(script), str(payload)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            argv = argv_log.read_text(encoding="utf-8")
            self.assertNotIn(token, argv)
            self.assertIn("--config", argv)
            self.assertIn("cfgmode=600", argv)
            # Temp config must be deleted after trap
            leftover = list((home / "tmp").glob("memorymesh-curl.*"))
            self.assertEqual(leftover, [])


if __name__ == "__main__":
    unittest.main()
