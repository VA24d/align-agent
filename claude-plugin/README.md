# stalecheck write-through for Claude Code

Claude Code's Edit tool notices when a file changed since Claude read it. A file changed through
Bash (`sed -i`, a script, a formatter, `git checkout`), or by you in another window, is not
checked: Claude keeps working from the copy it read, and a later edit by line number or by
remembered text can land in the wrong place. This plugin closes that gap.

- **Write-through.** After every tool call, each file Claude has seen that changed since gets its
  net diff appended to Claude's context, with current line numbers. If a changed definition is
  still used in other files, those uses are listed too. Nothing earlier in the conversation is
  rewritten, so the prompt cache is kept.
- **Line shifts.** Lines far below a change move without being shown. Each diff ends with a note
  such as `[LINE SHIFT: 'app.py': lines 15-40 of your last view are now lines 10-35 (-5)]`.
- **Line-number guard.** A `sed -i` addressed by line number, on a file Claude has seen, whose
  lines Claude has not seen numbered as they are now, is refused once. The refusal shows those
  lines as they stand, and Claude reissues the command with the right numbers.

## Install

```
/plugin marketplace add <owner>/stalecheck
/plugin install stalecheck-write-through@stalecheck
```

To try it from a checkout: `claude --plugin-dir tool/claude-plugin`.

Requires `python3` (3.8+) and, for the fast path, `git`. No packages to install.

## Settings (environment variables)

| Variable | Effect |
|---|---|
| `STALECHECK_SHIFTS=0` | no line-shift notes |
| `STALECHECK_GUARD=0` | no line-number guard |
| `STALECHECK_TIMING=1` | print the hook's time per call to stderr |

## What it costs

One `stat` per workspace file per tool call, and a hash only of files whose size or modification
time changed. In a 2,000-file repository a call takes well under 0.5 s; in our runs a call took
about 0.1 s including Python start-up. A diff adds tokens only when a file Claude has seen
changed outside its own Edit and Write tools, and stays under Claude Code's 10,000-character
limit for hook context.

## Limits

- Changes are noticed at the next tool call, not as they happen.
- The guard reads `sed -i` line addresses. Line edits made by `awk`, `perl -i` or inline scripts
  are covered by the diff and the shift note, not refused.
- Compaction or `/clear` empties the record of what Claude has seen, since Claude no longer holds
  those views.
- State is kept per session under `$CLAUDE_PLUGIN_DATA/sessions/`. Errors go to
  `$CLAUDE_PLUGIN_DATA/errors.log`, and the hook never blocks a session because of its own error.

The diff and the dependency notice are the code from the study behind this tool (`src/harness`,
copied into `scripts/_lib` by `tool/build.sh`).

## Licence

MIT.
