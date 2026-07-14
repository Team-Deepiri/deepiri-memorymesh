from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .providers.registry import (
    default_auto_providers,
    default_provider_globs,
    default_provider_paths,
)


def default_config_dir() -> Path:
    return Path.home() / ".config" / "deepiri-memorymesh"


def default_db_path() -> Path:
    """Canonical platform database path under the current process HOME."""
    return default_config_dir() / "memorymesh.db"


def default_config_path() -> Path:
    return default_config_dir() / "config.yaml"


# Import-time snapshots for backward-compatible module attributes.
DEFAULT_CONFIG_DIR = default_config_dir()
DEFAULT_CONFIG_PATH = default_config_path()
DEFAULT_DB_PATH = default_db_path()

# Retrieval defaults (T32). Not rewritten into existing YAML merely to add keys.
DEFAULT_RETRIEVAL_MODE = "auto"
DEFAULT_RETRIEVAL_EXACT_THRESHOLD = 200
DEFAULT_RETRIEVAL_CANDIDATE_LIMIT = 64


def normalize_db_path(value: str | Path) -> Path:
    """Normalize a configured database path for runtime use.

    Rules (T28):
    - ``~/...`` and ``~user/...`` expand via :meth:`Path.expanduser` using the
      process ``HOME`` (or platform equivalent).
    - Absolute paths remain absolute after expansion.
    - Relative paths stay relative to the process current working directory;
      they are **not** resolved against the config file directory and are **not**
      converted to absolute form via ``resolve()``.
    - Nonexistent paths are allowed; parents are created only when the store
      opens the database.
    - :meth:`Settings.load` does not rewrite YAML solely because the expanded
      runtime path differs from the stored string.
    """
    return Path(value).expanduser()


@dataclass(slots=True)
class Settings:
    db_path: Path = field(default_factory=default_db_path)
    embedding_backend: str = "fallback"
    providers: list[str] = field(default_factory=default_auto_providers)
    compression_max_chars: int = 6000
    compression_target_chars: int = 1200
    provider_paths: dict[str, str] = field(default_factory=default_provider_paths)
    provider_globs: dict[str, list[str]] = field(default_factory=default_provider_globs)
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE
    retrieval_exact_threshold: int = DEFAULT_RETRIEVAL_EXACT_THRESHOLD
    retrieval_candidate_limit: int = DEFAULT_RETRIEVAL_CANDIDATE_LIMIT
    # Host allowlist for `memorymesh pull api` (T38), merged with --allow-host.
    # Never a substitute for HTTPS/SSRF validation in api_pull.py.
    api_pull_allowed_hosts: list[str] = field(default_factory=list)
    # T37: default HTTP auth posture for `memorymesh serve` / supervised TUI
    # service when a CLI flag isn't explicitly given. "required" is the
    # secure-by-default value; "off" is for local development only.
    http_auth_mode: str = "required"
    # T33: optional path to the encryption key file. Never a secret itself —
    # only a filesystem reference. Env MEMORYMESH_ENCRYPTION_KEY also works.
    encryption_key_file: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        cfg_path = path or default_config_path()
        if not cfg_path.exists():
            cfg = cls()
            cfg.save(cfg_path)
            return cfg
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        defaults = cls()
        key_file_raw = raw.get("encryption_key_file")
        key_file = (
            normalize_db_path(key_file_raw) if key_file_raw not in (None, "") else None
        )
        return cls(
            db_path=normalize_db_path(raw.get("db_path", str(default_db_path()))),
            embedding_backend=raw.get("embedding_backend", "fallback"),
            providers=raw.get("providers") or defaults.providers,
            compression_max_chars=int(raw.get("compression_max_chars", 6000)),
            compression_target_chars=int(raw.get("compression_target_chars", 1200)),
            provider_paths=raw.get("provider_paths") or defaults.provider_paths,
            provider_globs=raw.get("provider_globs") or defaults.provider_globs,
            retrieval_mode=str(raw.get("retrieval_mode", defaults.retrieval_mode)),
            retrieval_exact_threshold=int(
                raw.get("retrieval_exact_threshold", defaults.retrieval_exact_threshold)
            ),
            retrieval_candidate_limit=int(
                raw.get("retrieval_candidate_limit", defaults.retrieval_candidate_limit)
            ),
            api_pull_allowed_hosts=list(
                raw.get("api_pull_allowed_hosts") or defaults.api_pull_allowed_hosts
            ),
            http_auth_mode=str(raw.get("http_auth_mode", defaults.http_auth_mode)),
            encryption_key_file=key_file,
        )

    def save(self, path: Path | None = None) -> None:
        cfg_path = path or default_config_path()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "db_path": str(self.db_path),
            "embedding_backend": self.embedding_backend,
            "providers": self.providers,
            "compression_max_chars": self.compression_max_chars,
            "compression_target_chars": self.compression_target_chars,
            "provider_paths": self.provider_paths,
            "provider_globs": self.provider_globs,
            "retrieval_mode": self.retrieval_mode,
            "retrieval_exact_threshold": self.retrieval_exact_threshold,
            "retrieval_candidate_limit": self.retrieval_candidate_limit,
            "api_pull_allowed_hosts": self.api_pull_allowed_hosts,
            "http_auth_mode": self.http_auth_mode,
        }
        if self.encryption_key_file is not None:
            # Store only the path reference — never key material.
            payload["encryption_key_file"] = str(self.encryption_key_file)
        cfg_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
