# Provider matrix

How Memory Mesh ingests provider/tool history today. Status labels describe **code and tests in this repository**, not live vendor certification.

Provider capability is defined by one canonical registry in
`deepiri_memorymesh/providers/registry.py`. Settings defaults, sync-auto,
provider-health, and parser routing all read that registry.

## Maturity legend

| Label | Meaning |
|---|---|
| **native** | Dedicated parser with fixture/unit coverage; eligible for automatic discovery |
| **generic-explicit** | Generic JSON/JSONL parser; available only via deliberate ingest/sync |
| **unsupported** | Placeholder name only; skipped by sync-auto; no verified native path |

**Important:** No provider in this matrix is live-provider-tested.

Evidence levels:

- **statically-implemented** — source present under `deepiri_memorymesh/providers/` or `integrations.py`
- **fixture-tested** — Python tests cover parser/sync behavior with synthetic inputs
- **generated-artifact runtime-tested** — compiled/executed generated plugin (OpenCode)
- **live-provider-tested** — verified against a real installed IDE/CLI session (**none claimed**)

## Automatic sync defaults

Fresh Settings auto-discover only native providers:

| Provider key | Default path | Default globs |
|---|---|---|
| `claude` | `~/.claude` | `**/*.json`, `**/*.jsonl` |
| `gemini` | `~/.config/google-gemini` | `**/*.json`, `**/*.jsonl` |
| `cursor` | `~/.cursor` | `**/*chat*.json`, `**/*chat*.jsonl`, `**/*conversation*.json*` |
| `opencode` | `~/.config/opencode` | `**/*.json`, `**/*.jsonl` |
| `continue` | `~/.continue` | `**/*.json`, `**/*.jsonl` |
| `aider` | `~/.aider` | `**/.aider.chat.history.md`, `**/.aider.input.history`, … |

Existing YAML may still list unsupported names (copilot, cline, openai, …). They load, but sync-auto reports them as **SKIPPED** rather than successful.

---

## Claude (`claude` / `anthropic`)

| | |
|---|---|
| **Kind** | native |
| **Parser** | `providers/claude.py` |
| **Formats** | JSON conversation objects; JSONL history lines |
| **Integration** | Bridge + hook + alias; Claude-only history fallback |
| **Evidence** | statically implemented; fixture-tested |
| **Limitations** | Not live-provider-tested. Non-Claude providers never read Claude history. |

## Cursor (`cursor`)

| | |
|---|---|
| **Kind** | native |
| **Parser** | `providers/cursor.py` |
| **Formats** | JSON chat objects; JSONL (malformed lines skipped) |
| **Evidence** | statically implemented; fixture-tested |
| **Limitations** | Depends on local Cursor export layout; not live-provider-tested. |

## Gemini (`gemini`)

| | |
|---|---|
| **Kind** | native |
| **Parser** | `providers/gemini.py` |
| **Evidence** | statically implemented; fixture-tested |
| **Limitations** | Export layouts vary; not live-provider-tested. |

## OpenCode (`opencode`)

| | |
|---|---|
| **Kind** | native |
| **Parser** | `providers/opencode.py` + native TypeScript plugin |
| **Integration** | Plugin at `~/.config/opencode/plugins/memorymesh.ts` |
| **Evidence** | statically implemented; fixture-tested; generated-artifact runtime-tested |
| **Limitations** | Idle ingest needs plugin + loopback service; not live-OpenCode tested. |

## Continue (`continue`)

| | |
|---|---|
| **Kind** | native |
| **Parser** | `providers/continue_dev.py` |
| **Evidence** | statically implemented; fixture-tested |
| **Limitations** | Session layouts differ across Continue versions; not live-tested. |

## Aider (`aider`)

| | |
|---|---|
| **Kind** | native |
| **Parser** | `providers/aider.py` |
| **Formats** | `.aider.chat.history.md` (primary); `.aider.input.history` (user-only, partial) |
| **Roles** | `user` / `assistant` / `system` (tool/status via `>` lines); recognized `####` headers retained in metadata |
| **Timestamps** | `# aider chat started at …` when present; otherwise fixed synthetic epoch + turn ordinal (never mutable mtime) |
| **Source keys** | Deterministic per turn for stable re-ingestion / append-only sync |
| **Integration** | `aider-memorymesh` wrapper |
| **Evidence** | statically implemented; fixture-tested |
| **Limitations** | Input-history is user-only; no invented assistant subtypes; not live-provider-tested. |

## Generic JSON / JSONL (`jsonl`)

| | |
|---|---|
| **Kind** | generic-explicit |
| **Parser** | `providers/base.py` `parse_generic_file` |
| **Auto-sync** | No (requires explicit ingest/sync) |
| **Limitations** | No vendor semantics. A generic message array is **not** a ChatGPT account export. |

## Unsupported placeholders

These names are **not** auto-discovered and have no verified native parser/fixtures:

`openai`, `copilot`, `cline`, `cody`, `perplexity`, `replit`, `ollama_local`, `lmstudio_local`, `llamacpp_local`

Old config entries remain loadable; sync-auto skips them with a visible diagnostic. Use `memorymesh ingest --provider jsonl` for deliberate generic files.

---

## Provider-specific fallback rules

- **Claude** bridges may fall back to `~/.claude/history.jsonl` when no transcript path is present.
- **Non-Claude providers never ingest Claude history.** Missing paths fail closed.
- OpenCode idle ingest uses the SDK + HTTP inline payload; it does not require Claude files.

See also [INTEGRATIONS.md](INTEGRATIONS.md) and the root [README.md](../README.md).
