# Using stalecheck write-through with your coding agent

`pip install git+https://github.com/VA24d/stalecheck@v0.1.0` provides the commands below; the engine itself uses only the standard library.
Each integration does three things where the agent allows it:

1. **after a tool call**, append the diff of every file the model has seen that changed since
   (by a shell command, a script, a formatter, or another person), with current line numbers, a
   note per stretch of lines that moved without being shown, and remaining uses in other files of
   changed definitions;
2. **record what the model reads**, through the agent's read tool or `cat`/`sed -n`/`head`/`grep -n`
   in the shell;
3. **before a shell command**, refuse once a `sed -i` on line numbers the model has not seen as they
   are now, and show those lines.

| Agent | 1 after | 2 reads | 3 refuse | How |
|---|---|---|---|---|
| Claude Code | yes | yes | yes | plugin: `/plugin marketplace add VA24d/stalecheck` |
| Codex CLI | yes | shell | yes | `codex/hooks.json` → `~/.codex/hooks.json` or `.codex/hooks.json`; trust it with `/hooks` |
| Gemini CLI | yes | yes | yes | merge `gemini/settings.json` into `.gemini/settings.json` |
| Cursor (IDE, CLI) | yes | yes | yes | `cursor/hooks.json` → `.cursor/hooks.json` |
| GitHub Copilot CLI | yes | yes | yes | `copilot/stalecheck.json` → `.github/hooks/` |
| OpenCode 1.x | yes | yes | yes | `opencode/stalecheck.js` → `.opencode/plugins/` |
| Cline | yes | yes | no | `cline/PostToolUse` → `.clinerules/hooks/` (its hooks can only cancel a whole task) |
| Goose, Antigravity CLI, others | yes | yes | yes | MCP: `mcp/README.md` (their hooks cannot add text after a tool) |
| Aider | n/a | | | re-reads the files in the chat on every request |

For agents with a hook system not listed here, `stalecheck-wt` takes one JSON event on stdin and
returns `{"context": ...}` or `{"deny": ...}` (`stalecheck-wt --help`).

## Notes per agent (documentation as of October 2026)

- **Codex CLI** ([hooks](https://developers.openai.com/codex/hooks)): the same hook schema as Claude
  Code, but unknown output fields fail the hook, so the adapter prints only Codex's fields. Codex has
  no read tool, so reads are taken from shell commands. Context over about 2,500 tokens is spilled to
  a file unless the handler sets `additionalContextLimit` (0 in the config here). Hooks outside a
  managed config must be trusted once with `/hooks`.
- **Gemini CLI** ([hooks reference](https://geminicli.com/docs/hooks/reference/)): `BeforeTool` /
  `AfterTool`; added context is appended to the tool result as `<hook_context>`. Google moved consumer
  users to the Antigravity CLI in June 2026; Antigravity's `PostToolUse` cannot add text, so use MCP there.
- **Cursor** ([hooks](https://cursor.com/docs/hooks)): `afterShellExecution` cannot add text, so the
  diff goes through `postToolUse`; the refusal uses `agent_message`, which the model reads. Cursor also
  imports Claude Code hooks from `.claude/settings.json`, but maps the refusal reason to the user, not
  the model, so prefer this native config. A permission hook that prints invalid JSON blocks the action;
  the adapter always prints valid JSON.
- **GitHub Copilot CLI** ([hooks reference](https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-hooks-reference)):
  `postToolUse` fires only on success, so `postToolUseFailure` is registered too; added context is
  capped at 10 KB. The cloud agent reads only `.github/hooks/*.json`.
- **OpenCode** ([plugins](https://opencode.ai/docs/plugins)): the V1 plugin appends to the tool output
  and throws to refuse. OpenCode 2 uses a different plugin API (`ctx.tool.hook`); V1 plugins do not run there.
- **Cline** (`.clinerules/hooks/README.md` in cline/cline): file hooks; `contextModification` reaches the
  model after the tool results. Refusing a single command needs a TypeScript plugin, available in the
  SDK and CLI but not yet in VS Code.

The adapters are `stalecheck/writethrough/hooks.py`; each is tested with the agent's documented event
payloads (`tests/test_claude_hook.py` in the research repository).
