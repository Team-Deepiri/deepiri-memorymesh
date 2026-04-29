# Provider Research Matrix

This matrix tracks practical ways to sync context/memory into MemoryMesh.

## Current strategy

- Default mode: local export/history file ingestion (`.json`, `.jsonl`)
- Optional mode: provider API pullers (future, token-gated)
- Local-model mode: filesystem logs and prompt history directories

## Providers

| Provider/Tool | Typical local data source | Current status | Notes |
|---|---|---|---|
| Claude Code / Anthropic exports | user export JSON files | scaffolded | Works via flexible JSON parser |
| Gemini | export JSON files | scaffolded | Supports role/text message arrays |
| ChatGPT/OpenAI exports | export JSON files | scaffolded | Supports multiple conversation key names |
| Cursor | `~/.cursor` history exports | scaffolded + auto-sync path | Bulk sync supported with `sync`/`sync-auto` |
| OpenCode | JSON/JSONL logs | scaffolded | Uses generic parser |
| GitHub Copilot Chat | local extension cache/export files | config placeholder | Add parser once stable format is confirmed |
| Continue.dev | `~/.continue` sessions/history | config placeholder | Good target for direct adapter |
| Aider | `.aider` chat logs | config placeholder | Add dedicated parser for command-rich turns |
| Cline / Roo Code | local extension logs | config placeholder | Add explicit tool-call extraction |
| Sourcegraph Cody | app cache/history | config placeholder | Verify export-friendly format first |
| Perplexity | export JSON (if available) | config placeholder | Usually manual export route |
| Replit Agent | workspace history exports | config placeholder | Depends on export access |
| Ollama local | wrapper logs + app history | config placeholder | Can integrate with local orchestrators |
| LM Studio local | app cache/history | config placeholder | Good for local-only memory sync |
| llama.cpp server | custom request/response logs | config placeholder | Add middleware logger adapter |

## Recommended next adapters

1. Cursor dedicated parser (`chat`, `composer`, and tool traces)
2. Continue.dev session parser
3. Aider transcript parser
4. Local model middleware logger parser (`ollama`, `llama.cpp`)

## Security notes

- Keep API keys out of ingested transcripts where possible.
- Use project namespaces to avoid cross-project leakage.
- Add encryption-at-rest if memory contains sensitive data.
