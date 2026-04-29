from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path.home() / ".config" / "deepiri-memorymesh"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"
DEFAULT_DB_PATH = DEFAULT_CONFIG_DIR / "memorymesh.db"


@dataclass(slots=True)
class Settings:
    db_path: Path = DEFAULT_DB_PATH
    embedding_backend: str = "fallback"
    providers: list[str] = field(
        default_factory=lambda: [
            "claude",
            "gemini",
            "openai",
            "cursor",
            "opencode",
            "jsonl",
            "copilot",
            "continue",
            "aider",
            "cline",
            "cody",
            "perplexity",
            "replit",
            "ollama_local",
            "lmstudio_local",
            "llamacpp_local",
        ]
    )
    compression_max_chars: int = 6000
    compression_target_chars: int = 1200
    provider_paths: dict[str, str] = field(
        default_factory=lambda: {
            "claude": "~/.claude",
            "gemini": "~/.config/google-gemini",
            "openai": "~/.config/openai",
            "cursor": "~/.cursor",
            "opencode": "~/.config/opencode",
            "copilot": "~/.config/github-copilot",
            "continue": "~/.continue",
            "aider": "~/.aider",
            "cline": "~/.cline",
            "cody": "~/.config/sourcegraph",
            "perplexity": "~/.config/perplexity",
            "replit": "~/.config/replit",
            "ollama_local": "~/.ollama",
            "lmstudio_local": "~/.cache/lm-studio",
            "llamacpp_local": "~/.local/share/llama.cpp",
        }
    )

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        cfg_path = path or DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            cfg = cls()
            cfg.save(cfg_path)
            return cfg
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return cls(
            db_path=Path(raw.get("db_path", str(DEFAULT_DB_PATH))),
            embedding_backend=raw.get("embedding_backend", "fallback"),
            providers=raw.get("providers") or cls().providers,
            compression_max_chars=int(raw.get("compression_max_chars", 6000)),
            compression_target_chars=int(raw.get("compression_target_chars", 1200)),
            provider_paths=raw.get("provider_paths") or cls().provider_paths,
        )

    def save(self, path: Path | None = None) -> None:
        cfg_path = path or DEFAULT_CONFIG_PATH
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "db_path": str(self.db_path),
            "embedding_backend": self.embedding_backend,
            "providers": self.providers,
            "compression_max_chars": self.compression_max_chars,
            "compression_target_chars": self.compression_target_chars,
            "provider_paths": self.provider_paths,
        }
        cfg_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
