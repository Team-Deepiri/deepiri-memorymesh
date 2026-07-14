"""Canonical provider capability registry (T19).

Single source of truth for parser routing, Settings defaults, sync-auto,
provider-health, CLI validation, and docs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ParserKind = Literal["native", "generic-explicit", "fallback-only", "unsupported"]
EvidenceLevel = Literal[
    "statically-implemented",
    "fixture-tested",
    "generated-artifact-runtime-tested",
    "live-provider-tested",
]


@dataclass(frozen=True, slots=True)
class ProviderCapability:
    name: str
    parser_kind: ParserKind
    automatic_discovery: bool
    default_path: str | None = None
    default_globs: tuple[str, ...] = ()
    accepted_formats: tuple[str, ...] = ()
    integration_support: str = "none"
    evidence: tuple[EvidenceLevel, ...] = ()
    limitations: str = ""
    aliases: tuple[str, ...] = ()
    parser_symbol: str | None = None


# Native parsers with fixture/unit coverage (Aider upgraded in T20).
_NATIVE: list[ProviderCapability] = [
    ProviderCapability(
        name="claude",
        parser_kind="native",
        automatic_discovery=True,
        default_path="~/.claude",
        default_globs=("**/*.json", "**/*.jsonl"),
        accepted_formats=("json", "jsonl"),
        integration_support="bridge+hooks",
        evidence=("statically-implemented", "fixture-tested"),
        limitations="Not live-provider-tested. Claude-only history fallback.",
        aliases=("anthropic",),
        parser_symbol="parse_claude_file",
    ),
    ProviderCapability(
        name="cursor",
        parser_kind="native",
        automatic_discovery=True,
        default_path="~/.cursor",
        default_globs=(
            "**/*chat*.json",
            "**/*chat*.jsonl",
            "**/*conversation*.json*",
        ),
        accepted_formats=("json", "jsonl"),
        integration_support="bridge+hooks",
        evidence=("statically-implemented", "fixture-tested"),
        limitations="Depends on local Cursor export layout; not live-provider-tested.",
        parser_symbol="parse_cursor_file",
    ),
    ProviderCapability(
        name="gemini",
        parser_kind="native",
        automatic_discovery=True,
        default_path="~/.config/google-gemini",
        default_globs=("**/*.json", "**/*.jsonl"),
        accepted_formats=("json", "jsonl"),
        integration_support="bridge+alias",
        evidence=("statically-implemented", "fixture-tested"),
        limitations="Export layouts vary; not live-provider-tested.",
        parser_symbol="parse_gemini_file",
    ),
    ProviderCapability(
        name="opencode",
        parser_kind="native",
        automatic_discovery=True,
        default_path="~/.config/opencode",
        default_globs=("**/*.json", "**/*.jsonl"),
        accepted_formats=("json", "jsonl"),
        integration_support="native-plugin",
        evidence=(
            "statically-implemented",
            "fixture-tested",
            "generated-artifact-runtime-tested",
        ),
        limitations="Idle ingest needs plugin + loopback service; not live-OpenCode tested.",
        parser_symbol="parse_opencode_file",
    ),
    ProviderCapability(
        name="continue",
        parser_kind="native",
        automatic_discovery=True,
        default_path="~/.continue",
        default_globs=("**/*.json", "**/*.jsonl"),
        accepted_formats=("json", "jsonl"),
        integration_support="bridge+hooks",
        evidence=("statically-implemented", "fixture-tested"),
        limitations="Session layouts differ across Continue versions; not live-tested.",
        parser_symbol="parse_continue_file",
    ),
    ProviderCapability(
        name="aider",
        parser_kind="native",
        automatic_discovery=True,
        default_path="~/.aider",
        default_globs=(
            "**/.aider.chat.history.md",
            "**/.aider.input.history",
            "**/aider.chat.history.md",
            "**/*.aider.chat.history.md",
        ),
        accepted_formats=("aider-chat-md", "aider-input-history"),
        integration_support="wrapper",
        evidence=("statically-implemented", "fixture-tested"),
        limitations=(
            "Parses evidenced Aider chat-history markdown and partial input-history; "
            "not live-provider-tested."
        ),
        parser_symbol="parse_aider_file",
    ),
]

_GENERIC: list[ProviderCapability] = [
    ProviderCapability(
        name="jsonl",
        parser_kind="generic-explicit",
        automatic_discovery=False,
        default_path=None,
        default_globs=("**/*.jsonl",),
        accepted_formats=("json", "jsonl"),
        integration_support="explicit-ingest",
        evidence=("statically-implemented", "fixture-tested"),
        limitations="No vendor semantics; use only with deliberate generic ingestion.",
        aliases=("json",),
        parser_symbol="parse_generic_file",
    ),
]

_UNSUPPORTED_NAMES = (
    "openai",
    "copilot",
    "cline",
    "cody",
    "perplexity",
    "replit",
    "ollama_local",
    "lmstudio_local",
    "llamacpp_local",
)

# Historical default paths retained only for classification/diagnostics when
# old YAML still lists these keys. They are NOT used for automatic discovery.
_UNSUPPORTED_PATH_HINTS: dict[str, str] = {
    "openai": "~/.config/openai",
    "copilot": "~/.config/github-copilot",
    "cline": "~/.cline",
    "cody": "~/.config/sourcegraph",
    "perplexity": "~/.config/perplexity",
    "replit": "~/.config/replit",
    "ollama_local": "~/.ollama",
    "lmstudio_local": "~/.cache/lm-studio",
    "llamacpp_local": "~/.local/share/llama.cpp",
}


def _unsupported(name: str) -> ProviderCapability:
    return ProviderCapability(
        name=name,
        parser_kind="unsupported",
        automatic_discovery=False,
        default_path=_UNSUPPORTED_PATH_HINTS.get(name),
        default_globs=(),
        accepted_formats=(),
        integration_support="none",
        evidence=(),
        limitations=(
            "No verified native parser or fixtures. Skipped by sync-auto. "
            "Explicit generic JSON/JSONL ingest may still be used via provider=jsonl."
        ),
        parser_symbol=None,
    )


PROVIDERS: dict[str, ProviderCapability] = {}
for cap in _NATIVE + _GENERIC:
    PROVIDERS[cap.name] = cap
    for alias in cap.aliases:
        PROVIDERS[alias] = cap
for name in _UNSUPPORTED_NAMES:
    PROVIDERS[name] = _unsupported(name)


def get_provider(name: str) -> ProviderCapability:
    key = name.strip().lower()
    if key in PROVIDERS:
        return PROVIDERS[key]
    # Unknown names are treated as unsupported placeholders.
    return ProviderCapability(
        name=key,
        parser_kind="unsupported",
        automatic_discovery=False,
        limitations="Unknown provider; not auto-discovered.",
    )


def list_providers() -> list[ProviderCapability]:
    """Unique capability records (aliases collapsed to primary name)."""
    seen: set[str] = set()
    out: list[ProviderCapability] = []
    for cap in list(_NATIVE) + list(_GENERIC) + [_unsupported(n) for n in _UNSUPPORTED_NAMES]:
        if cap.name in seen:
            continue
        seen.add(cap.name)
        out.append(cap)
    return out


def default_auto_providers() -> list[str]:
    """Provider keys enabled for automatic discovery in fresh Settings."""
    return [c.name for c in _NATIVE if c.automatic_discovery]


def default_provider_paths() -> dict[str, str]:
    out: dict[str, str] = {}
    for cap in _NATIVE:
        if cap.default_path and cap.automatic_discovery:
            out[cap.name] = cap.default_path
    return out


def default_provider_globs() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cap in _NATIVE:
        if cap.default_globs and cap.automatic_discovery:
            out[cap.name] = list(cap.default_globs)
    # Explicit generic remains available when configured.
    out["jsonl"] = list(PROVIDERS["jsonl"].default_globs)
    return out


def native_parser_map() -> dict[str, str]:
    """Compatibility map: provider key → parser symbol name."""
    out: dict[str, str] = {}
    for key, cap in PROVIDERS.items():
        if cap.parser_kind == "native" and cap.parser_symbol:
            out[key] = cap.parser_symbol
    return out


def normalize_provider_key(name: str) -> str:
    """Return the canonical primary provider name (aliases collapse)."""
    return get_provider(name).name


def allows_automatic_discovery(name: str) -> bool:
    cap = get_provider(name)
    return bool(cap.automatic_discovery and cap.parser_kind == "native")


def allows_native_integration(name: str) -> bool:
    """True when install-native / bridge artifacts are permitted."""
    cap = get_provider(name)
    if cap.parser_kind == "native" and cap.integration_support not in {"", "none"}:
        return True
    # Explicit generic ingest bridges remain available for jsonl.
    if cap.parser_kind == "generic-explicit":
        return True
    return False


def native_integration_provider_names() -> list[str]:
    """Primary names eligible for ``install-native`` / ``install-native-all``."""
    return [
        c.name
        for c in list_providers()
        if c.parser_kind == "native" and c.integration_support not in {"", "none"}
    ]


@dataclass(slots=True)
class SyncAutoProviderOutcome:
    provider: str
    classification: str
    skipped: bool
    reason: str = ""
    processed: int = 0
    failed: int = 0
    inserted: int = 0
    path: str = ""
