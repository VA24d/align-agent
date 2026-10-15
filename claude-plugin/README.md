# ALIGN write-through for Claude Code

Claude Code tells Claude about a changed file only if Claude last wrote it with Edit or Write, and
then reports even Claude's own Bash changes as outside edits
([anthropics/claude-code#81503](https://github.com/anthropics/claude-code/issues/81503)). A file
Claude only read and then changed through Bash (`sed -i`, a script, a formatter, `git checkout`)
is not checked: Claude keeps working from the copy it read, and a later edit by line number or by
remembered text can land in the wrong place. This plugin closes that gap.

- **Write-through.** After every tool call, each file Claude has seen that changed since gets its
  net diff appended to Claude's context, with current line numbers. If a changed definition is
  still used in other files, those uses are listed too. Nothing earlier in the conversation is
  rewritten, so the prompt cache is kept.
- **Line numbers.** Lines far below a change move without being shown. Each diff ends with a note
  that counts from the last time Claude saw the whole file, so offsets from several edits add up:
  `[LINE NUMBERS: since your last full view of 'app.py': lines 15-40 are now 10-35 (-5)]`.
  In a 60-run test with Claude Sonnet 5 the note kept accuracy at the plain diff's level and cut
  re-reads of the file (2.0 against 2.4 per run); a version counted from each previous diff did worse.
- **Line-number guard.** A `sed -i` addressed by line number, on a file Claude has seen, whose
  lines Claude has not seen numbered as they are now, is refused once. The refusal shows those
  lines as they stand, and Claude reissues the command with the right numbers.

## Install

```
/plugin marketplace add VA24d/align-agent
/plugin install align-write-through@align-agent
```

To try it from a checkout: `claude --plugin-dir claude-plugin` (from the repository root).

Requires `python3` (3.10+) and, for the fast path, `git`. No packages to install.

## Settings (environment variables)

| Variable | Effect |
|---|---|
| `ALIGN_SHIFTS=0` | no line-number notes |
| `ALIGN_GUARD=0` | no line-number guard |
| `ALIGN_TIMING=1` | print the hook's time per call to stderr |

## What it costs

One `stat` per workspace file per tool call, and a hash only of files whose size or modification
time changed. In a 2,000-file repository a call takes well under 0.5 s; in our runs a call took
about 0.1 s including Python start-up. A diff adds tokens only when a file Claude has seen
changed outside its own Edit and Write tools, and stays under Claude Code's 10,000-character
limit for hook context.

## Limits

- Changes are noticed at the next tool call, not as they happen.
- The guard reads `sed -i` line addresses. Line edits made by `awk`, `perl -i` or inline scripts
  are covered by the diff and the line-number note, not refused.
- Compaction or `/clear` empties the record of what Claude has seen, since Claude no longer holds
  those views.
- State is kept per session under `$CLAUDE_PLUGIN_DATA/sessions/`. It holds a copy of each file the model viewed (user-only permissions); files named like keys, `.env` files and credential stores are never copied or diffed, copies no longer needed are deleted after each call, and sessions idle for 7 days (`ALIGN_KEEP_DAYS`) are deleted. Errors go to
  `$CLAUDE_PLUGIN_DATA/errors.log`, and the hook never blocks a session because of its own error.

The diff and the dependency notice are the code from the study behind this tool (`src/harness`,
copied into `scripts/_lib` by `tool/build.sh`).

## Licence

MIT.
