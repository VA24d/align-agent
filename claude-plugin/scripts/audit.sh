#!/usr/bin/env bash
# Count stale edits in this machine's own Claude Code and Codex session logs.
# Prints one summary line per folder (counts only: no paths, prompts or code), then which of your own
# project folders and sessions hold the most stale edits (names stay on this machine). Nothing is uploaded.
#
# It needs the `align-agent` command. If that is not installed, it installs it once, from PyPI, into its
# own virtual environment (~/.cache/align-agent/venv) and keeps it up to date. Errors are shown, not hidden.
# Works with Python 3.10 or later as python3, python or `py -3` (Windows, in Git Bash).

CACHE="${ALIGN_CACHE:-$HOME/.cache/align-agent}"
VENV="$CACHE/venv"

find_python() {
  for c in python3.13 python3.12 python3.11 python3.10 python3 python "py -3"; do
    if $c -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  return 1
}

align_command() {
  if command -v align-agent >/dev/null 2>&1; then echo "align-agent"; return 0; fi
  for b in "$VENV/bin/align-agent" "$VENV/Scripts/align-agent.exe" "$VENV/Scripts/align-agent"; do
    if [ -x "$b" ]; then
      # keep it current; --quiet and the version check make this a no-op when it is
      "$(dirname "$b")/python" -m pip install --quiet --upgrade --disable-pip-version-check align-agent >&2 || true
      echo "$b"; return 0
    fi
  done
  if command -v uvx >/dev/null 2>&1; then echo "uvx --quiet align-agent"; return 0; fi
  if command -v pipx >/dev/null 2>&1; then echo "pipx run align-agent"; return 0; fi
  py="$(find_python)" || { echo "align-agent: no Python 3.10 or later found (tried python3, python, py -3). Install Python from https://www.python.org/downloads/ and run this again." >&2; return 1; }
  echo "align-agent: installing it from PyPI into $VENV (first run only)..." >&2
  mkdir -p "$CACHE" && $py -m venv "$VENV" >&2 || { echo "align-agent: could not create a virtual environment in $VENV" >&2; return 1; }
  for b in "$VENV/bin/python" "$VENV/Scripts/python.exe" "$VENV/Scripts/python"; do
    [ -x "$b" ] && "$b" -m pip install --quiet --upgrade --disable-pip-version-check align-agent >&2 && break
  done
  for b in "$VENV/bin/align-agent" "$VENV/Scripts/align-agent.exe" "$VENV/Scripts/align-agent"; do
    [ -x "$b" ] && { echo "$b"; return 0; }
  done
  echo "align-agent: the install failed; see the messages above" >&2
  return 1
}

AA="$(align_command)" || exit 1
found=0
for d in "$HOME/.claude/projects" "$HOME/.codex/sessions"; do
  [ -d "$d" ] || continue
  found=1
  printf '%s: ' "$(basename "$(dirname "$d")")"
  $AA analyze "$d" --quiet --top 5 -j 4 2>"$CACHE/last_error.txt" | sed '/^$/d'
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "(failed: $(tail -n 3 "$CACHE/last_error.txt" 2>/dev/null | tr '\n' ' '))"; }
done
[ "$found" = 1 ] || echo "no Claude Code or Codex session folders found (looked in ~/.claude/projects and ~/.codex/sessions)"
