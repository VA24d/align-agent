# ALIGN: Agent Line-Index Guard and Notices

A coding agent edits files through what it last saw of them, and its own edits make those views
out of date. A `sed -i`, a script, a formatter or `git checkout` changes a file the agent read, and
nothing tells it; a later edit by line number or by remembered text then lands on the wrong lines,
is applied twice, or misses. ALIGN keeps the agent's view current while it works, and finds
these edits in recorded agent trajectories.

## Keep your agent's view current

After every tool call, ALIGN appends to what the model sees the diff of each file it has seen
that changed since, with current line numbers, a note for each stretch of lines that moved without
being shown, and the remaining uses in other files of any definition a change removed or
re-signatured. Before an in-place edit by line number (`sed`, `perl`, `awk`, `ed`, `ex`) on lines the model has
not seen numbered as they are now, it refuses
the command once and shows those lines. Nothing earlier in the conversation is rewritten, so the
prompt cache is kept.

```bash
pip install git+https://github.com/VA24d/align-agent@v0.4.0   # the write-through engine itself uses only the standard library
```

| Agent | Set up |
|---|---|
| Claude Code | `/plugin marketplace add VA24d/align-agent`, then `/plugin install align-write-through@align-agent` |
| Codex CLI, Gemini CLI, Cursor, GitHub Copilot CLI, OpenCode, Cline | the config for your agent in [`integrations/`](integrations/README.md) |
| Any MCP client (Goose, Antigravity CLI, IDEs, ...) | `align-agent-mcp` as an MCP server, with the built-in shell turned off ([`integrations/mcp`](integrations/mcp/README.md)) |
| Your own harness | `align-agent-wt`: one JSON event on stdin, `{"context": ...}` or `{"deny": ...}` back; or `from align_agent.writethrough import Session` |

In a 2,000-file repository a call takes well under 0.5 s; in Claude Code a call took about 0.1 s,
Python start-up included.

To compute diffs, ALIGN keeps a copy of each file the model viewed in its state folder
(`~/.cache/align-agent`, or the plugin's data folder), readable only by you. Files named like
keys, `.env` files and credential stores are never copied or diffed; copies no longer needed are
deleted after each call, and sessions idle for 7 days are deleted (`ALIGN_KEEP_DAYS`).

## Find stale edits in trajectories

```bash
pip install git+https://github.com/VA24d/align-agent@v0.4.0
align-agent analyze examples/teleport_control.json examples/teleport_diff.json
```

Or, with nothing to install or build (image for amd64 and arm64, examples included):

```bash
docker run --rm ghcr.io/va24d/align-agent:0.4.0 analyze /opt/align-agent/examples/teleport_control.json
docker run --rm -v "$PWD:/work" ghcr.io/va24d/align-agent:0.4.0 analyze /work/my_trajectories/
```

## `align-agent analyze`

```
align-agent analyze PATH [PATH ...] [--format F] [--cwd DIR] [--json OUT] [--quiet]
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
| `claudecode` | Claude Code session logs (`~/.claude/projects/*/*.jsonl`) |
| `codex` | Codex CLI session rollouts (`~/.codex/sessions`) |

A `.jsonl` file is a Claude Code or Codex session, told apart by the shape of its lines, and a directory is searched for `*.json` and `*.jsonl`. To audit your own sessions:

```bash
align-agent analyze ~/.claude/projects/-home-me-myrepo/      # every session of one project, and its subagents
align-agent analyze ~/.codex/sessions/2026/10/ --json codex.json
```

Paths resolve against the working directory the log records. In Claude Code, `Read` is a view of the lines it returned, `Edit` and `MultiEdit` are edits anchored on quoted text, `Write` is a whole-file write, and `Bash` goes through the shell parser. Claude Code's changed-file notices count as views. A subagent shares the workspace but not the context, so in the parent session its writes are writes the parent did not see; its own transcript is analyzed as a separate trajectory. In Codex, `apply_patch` hunks are anchored edits, `*** Add File` is a whole-file write, and shell commands (including those a code-mode `exec` script passes to `exec_command` as literals) go through the shell parser. A context compaction starts over: views from before it no longer count. `--json` gives per-session counts of every line and tool the adapters did not map.

Each command is parsed into file operations: views (`cat`, `sed -n`, `grep -n`, editor views, file paths quoted in compiler and test output), anchored edits (`sed` by line or pattern, `str_replace`, Python `.replace`, patches) and whole-file writes. Each view is bound to the version of the file it showed. An anchored edit to a file whose last view no longer matches the file is a **stale edit**. For each one, the output gives the step that held the view, whether that view came from the shell or an editor tool, which unseen writes changed the file, and a class:

- `MATERIALIZED-M3`: the edit went wrong because of the stale view (applied twice, or a line anchor that the unseen writes moved).
- `BENIGN-FORWARDED`, `BENIGN-DISJOINT`: the unseen change was accounted for, or did not touch what the edit relied on.
- `FAILED-OWN-WRITE`, `FAILED-UNRELATED`, `SPURIOUS`: the edit failed for its own reasons, or the unseen write changed nothing.
- `INDETERMINATE`: the transcript does not show enough to decide.

The summary reports the stale-edit rate over anchored edits to viewed files, with a 95% Clopper–Pearson interval. `--json` writes every edit.

## `align-agent run`

```
align-agent run --repo DIR --task TEXT_OR_FILE --model NAME --base-url URL \
               [--arm control|viewport|diff|lean|dependency] [--image IMAGE] [--out trajectory.json]
```

Needs the `[run]` extra: `pip install 'align-agent[run] @ git+https://github.com/VA24d/align-agent@v0.4.0'`. Runs mini-swe-agent 2.4.6 against any OpenAI-compatible endpoint. The API key is read from `$OPENAI_API_KEY`, or from the variable that `--api-key-env` names. After each command, the hook finds the files the command changed by content hash and appends one of these to that command's output:

- `viewport`: a numbered window around each change.
- `diff`: the net diff since the agent last saw the file.
- `lean`: the same diff, skipped for writes the agent already knows (a whole file it supplied, a file the command printed).
- `dependency`: the diff, plus the remaining uses in other files of any definition the write removed or re-signatured.

Nothing earlier in the conversation is rewritten, so the prompt stays cacheable. With `--image`, commands run in that container with the repository mounted. Without it, use `--no-sandbox` only inside a container. The trajectory is written in mini-swe-agent's format, so `analyze` reads it back.

## What the numbers mean

The analyzer is the scorer and classifier of the study behind this tool, unchanged. On the published trajectories it was built for, an independent hand audit of every stale edit agreed with a second blind labeller at κ = 0.73 on the classes. The `MATERIALIZED` class comes from rules, not execution, and errs both ways. Treat a `MATERIALIZED` label as a pointer to a step worth reading, not as a verdict.

## Licence

MIT. The bundled examples are trajectories from our own runs.
