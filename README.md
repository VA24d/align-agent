# stalecheck

A coding agent edits files through what it last saw of them, and its own edits make those views
out of date. A `sed -i`, a script, a formatter or `git checkout` changes a file the agent read, and
nothing tells it; a later edit by line number or by remembered text then lands on the wrong lines,
is applied twice, or misses. `stalecheck` keeps the agent's view current while it works, and finds
these edits in recorded agent trajectories.

## Keep your agent's view current

After every tool call, stalecheck appends to what the model sees the diff of each file it has seen
that changed since, with current line numbers, a note for each stretch of lines that moved without
being shown, and the remaining uses in other files of any definition a change removed or
re-signatured. Before a `sed -i` on line numbers the model has not seen as they are now, it refuses
the command once and shows those lines. Nothing earlier in the conversation is rewritten, so the
prompt cache is kept.

```bash
pip install git+https://github.com/VA24d/stalecheck   # the write-through engine needs no other package
```

| Agent | Set up |
|---|---|
| Claude Code | `/plugin marketplace add VA24d/stalecheck`, then `/plugin install stalecheck-write-through@stalecheck` |
| Codex CLI, Gemini CLI, Cursor, GitHub Copilot CLI, OpenCode, Cline | the config for your agent in [`integrations/`](integrations/README.md) |
| Any MCP client (Goose, Antigravity CLI, IDEs, ...) | `stalecheck-mcp` as an MCP server, with the built-in shell turned off ([`integrations/mcp`](integrations/mcp/README.md)) |
| Your own harness | `stalecheck-wt`: one JSON event on stdin, `{"context": ...}` or `{"deny": ...}` back; or `from stalecheck.writethrough import Session` |

In a 2,000-file repository a call takes well under 0.5 s; in Claude Code a call took about 0.1 s,
Python start-up included.

## Find stale edits in trajectories

```bash
pip install git+https://github.com/VA24d/stalecheck
stalecheck analyze examples/teleport_control.json examples/teleport_diff.json
```

## `stalecheck analyze`

```
stalecheck analyze PATH [PATH ...] [--format F] [--cwd DIR] [--json OUT] [--quiet]
```

Reads trajectory files, or directories of them. The format is detected:

| `--format` | Source |
|---|---|
| `mini` | mini-swe-agent trajectories (text actions, `<returncode>` observations) |
| `sweagent` | SWE-agent `.traj` |
| `openhands` | OpenHands completion logs and event streams |
| `terminus` | Terminus 2 episodes |
| `atif` | Harbor ATIF trials, e.g. the DeepSWE leaderboard |
| `toolcalls` | OpenAI-style messages with a `bash` tool |

Each command is parsed into file operations: views (`cat`, `sed -n`, `grep -n`, editor views, file paths quoted in compiler and test output), anchored edits (`sed` by line or pattern, `str_replace`, Python `.replace`, patches) and whole-file writes. Each view is bound to the version of the file it showed. An anchored edit to a file whose last view no longer matches the file is a **stale edit**. For each one, the output gives the step that held the view, whether that view came from the shell or an editor tool, which unseen writes changed the file, and a class:

- `MATERIALIZED-M3`: the edit went wrong because of the stale view (applied twice, or a line anchor that the unseen writes moved).
- `BENIGN-FORWARDED`, `BENIGN-DISJOINT`: the unseen change was accounted for, or did not touch what the edit relied on.
- `FAILED-OWN-WRITE`, `FAILED-UNRELATED`, `SPURIOUS`: the edit failed for its own reasons, or the unseen write changed nothing.
- `INDETERMINATE`: the transcript does not show enough to decide.

The summary reports the stale-edit rate over anchored edits to viewed files, with a 95% Clopper–Pearson interval. `--json` writes every edit.

## `stalecheck run`

```
stalecheck run --repo DIR --task TEXT_OR_FILE --model NAME --base-url URL \
               [--arm control|viewport|diff|lean|dependency] [--image IMAGE] [--out trajectory.json]
```

Needs the `[run]` extra: `pip install 'stalecheck[run] @ git+https://github.com/VA24d/stalecheck'`. Runs mini-swe-agent 2.4.6 against any OpenAI-compatible endpoint. The API key is read from `$OPENAI_API_KEY`, or from the variable that `--api-key-env` names. After each command, the hook finds the files the command changed by content hash and appends one of these to that command's output:

- `viewport`: a numbered window around each change.
- `diff`: the net diff since the agent last saw the file.
- `lean`: the same diff, skipped for writes the agent already knows (a whole file it supplied, a file the command printed).
- `dependency`: the diff, plus the remaining uses in other files of any definition the write removed or re-signatured.

Nothing earlier in the conversation is rewritten, so the prompt stays cacheable. With `--image`, commands run in that container with the repository mounted. Without it, use `--no-sandbox` only inside a container. The trajectory is written in mini-swe-agent's format, so `analyze` reads it back.

## What the numbers mean

The analyzer is the scorer and classifier of the study behind this tool, unchanged. On the published trajectories it was built for, an independent hand audit of every stale edit agreed with a second blind labeller at κ = 0.73 on the classes. The `MATERIALIZED` class comes from rules, not execution, and errs both ways. Treat a `MATERIALIZED` label as a pointer to a step worth reading, not as a verdict.

## Licence

MIT. The bundled examples are trajectories from our own runs.
