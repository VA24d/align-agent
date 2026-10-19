#!/usr/bin/env bash
# Count stale edits in this machine's own Claude Code and Codex session logs.
# Prints one summary line per folder (counts only: no paths, prompts or code), then which of your own
# project folders and sessions hold the most stale edits (names stay on this machine). Nothing is uploaded.
# Uses `align-agent` if it is installed, else `uvx align-agent` (no install step) or `pipx run`.
run() {
  if command -v align-agent >/dev/null 2>&1; then align-agent "$@"
  elif command -v uvx >/dev/null 2>&1; then uvx --quiet align-agent "$@"
  elif command -v pipx >/dev/null 2>&1; then pipx run align-agent "$@"
  else echo "install align-agent first: pip install align-agent (or install uv: https://docs.astral.sh/uv/)" >&2; return 1
  fi
}
found=0
for d in "$HOME/.claude/projects" "$HOME/.codex/sessions"; do
  [ -d "$d" ] || continue
  found=1
  printf '%s: ' "$(basename "$(dirname "$d")")"
  run analyze "$d" --quiet --top 5 -j 4 2>/dev/null | sed '/^$/d' 
done
[ "$found" = 1 ] || echo "no Claude Code or Codex session folders found"
