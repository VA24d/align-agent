---
description: Count stale edits in your own Claude Code and Codex sessions (counts only, nothing is uploaded)
allowed-tools: Bash(bash:*)
---
Run `bash "${CLAUDE_PLUGIN_ROOT}/scripts/audit.sh"` with the Bash tool. It prints one summary line per
session folder: counts only. Reply with those lines unchanged, then one sentence giving the share of
anchored edits that were stale and how many came through the shell. Do not open or quote any session file.
