# Provider Storage Paths

Device scan targets these locations by default.

## Claude Code

- **Sessions:** `~/.claude/projects/<project-hash>/*.jsonl`
- **History index:** `~/.claude/history.jsonl`
- **Config override:** `CLAUDE_CONFIG_DIR`

Docs: https://code.claude.com/docs/en/agent-sdk/sessions

## Cursor

- **Global DB:** `<Cursor User>/globalStorage/state.vscdb` — `bubbleId:*`, `composerData:*`
- **Workspace DBs:** `<Cursor User>/workspaceStorage/<id>/state.vscdb`
- **Transcripts:** `~/.cursor/projects/<sanitized-path>/agent-transcripts/*.txt`

| OS | Cursor User directory |
|----|------------------------|
| Linux | `~/.config/Cursor/User` |
| macOS | `~/Library/Application Support/Cursor/User` |
| Windows | `%APPDATA%\Cursor\User` |

## OpenCode

- **Data:** `~/.local/share/opencode/` (or `OPENCODE_DATA_DIR`)
- **Config:** `~/.config/opencode/`
- **In-repo:** `.opencode/storage/` when inside a git project

Docs: https://opencode.ai/docs/troubleshooting/
