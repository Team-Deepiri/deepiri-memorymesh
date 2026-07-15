# U-Data Packaging

Portable memory packages for moving context between Claude Code, Cursor, and OpenCode.

## Storage locations (researched)

| Tool | Where data lives |
|------|------------------|
| **Claude Code** | `~/.claude/projects/<hash>/*.jsonl` (transcripts), `~/.claude/history.jsonl` (index). Override with `CLAUDE_CONFIG_DIR`. |
| **Cursor** | `~/.config/Cursor/User/globalStorage/state.vscdb` (messages), `workspaceStorage/*/state.vscdb` (per-project index). Linux path; macOS uses `~/Library/Application Support/Cursor/User`. |
| **OpenCode** | `~/.local/share/opencode/` (sessions), project `./.opencode/storage/` in git repos. Override with `OPENCODE_DATA_DIR`. |

## CLI quick start

```bash
# See what exists on this machine
memorymesh scan

# Scan + ingest into local memory DB
memorymesh pull -p myproject

# One command: scan device → ingest → export portable package
memorymesh package build -p myproject -o ./my-udata.tar.gz

# On another machine
memorymesh package import ./my-udata.tar.gz -p myproject

# Export transfer JSON for a specific target tool
memorymesh package transfer -p myproject --from claude --to cursor -o ./to-cursor.json
```

## Package format

`memorymesh-u-data-v1` JSON (or `.tar.gz` containing `udata.json`):

- `manifest` — hostname, platform, provider stats, source paths
- `messages` — normalized messages from all ingested providers
- `summaries` — optional compressed conversation summaries

## Tips

- Close Cursor before scanning SQLite DBs for consistent reads (WAL files are copied when present).
- Use project namespaces (`-p`) to avoid mixing work contexts.
- Run `memorymesh package build --compress` to include summaries in the export.
