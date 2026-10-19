"""Claude Code adapter: SessionStart, PreToolUse and PostToolUse hooks.

Claude Code sends each event as JSON on stdin. PostToolUse text goes back as
hookSpecificOutput.additionalContext (shown to the model next to the tool result, at most
10,000 characters); a PreToolUse refusal as permissionDecision "deny" with a reason the model
reads. Claude Code's own Read records a view; Edit/Write/MultiEdit/NotebookEdit are changes the
model made itself. State goes to $CLAUDE_PLUGIN_DATA, which Claude Code sets for plugins.
"""

import json
import os
import re
import sys
import time

from .engine import Session, default_state_dir

READ_TOOLS = {"Read"}
EDIT_TOOLS = {"Edit", "MultiEdit", "NotebookEdit"}
WRITE_TOOLS = {"Write"}  # the model typed the whole file
READ_LIMIT = 2000  # lines Read shows when given no offset or limit
SHELL_TOOLS = {"Bash", "PowerShell"}


def session_key(payload):
    """A subagent has its own context, so its own views: hooks fired inside one carry its agent_id."""
    key = payload.get("session_id") or "default"
    if payload.get("agent_id"):
        key += "-" + str(payload["agent_id"])
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


def handle(payload):
    """The hook's JSON reply for one event, or None."""
    event = payload.get("hook_event_name")
    root = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
    # Claude Code re-sends the changed lines of a file its editor wrote last after a shell change
    cwd = payload.get("cwd")  # the shell's directory, which `cd` and worktrees move; the root stays
    with Session(root, session_key(payload), host_echoes=True, read_limit=READ_LIMIT) as s:
        if event == "SessionStart":
            # resume keeps what the model saw; a new, cleared, compacted or forked context does not
            s.start(fresh=payload.get("source") in ("startup", "clear", "compact", "fork"))
            return None
        tool = payload.get("tool_name") or ""
        inp = payload.get("tool_input") or {}
        if event == "PreToolUse":
            reason = s.before(inp.get("command", ""), cwd=cwd) if tool in SHELL_TOOLS else None
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                           "permissionDecisionReason": reason}} if reason else None
        if event not in ("PostToolUse", "PostToolUseFailure"):  # a failed tool can still have changed files
            return None
        failed = event == "PostToolUseFailure"  # nothing was shown or written by the tool itself
        path = inp.get("file_path") or inp.get("notebook_path") or inp.get("path")
        read = path if tool in READ_TOOLS and not failed else None
        rng = None
        if read and (inp.get("offset") or inp.get("limit")):
            lo = int(inp.get("offset") or 1)
            rng = (lo, lo + int(inp.get("limit") or 2000) - 1)
        command = inp.get("command", "") if tool in SHELL_TOOLS else ""
        resp = payload.get("tool_response", payload.get("error"))
        output = "\n".join(str(resp.get(k) or "") for k in ("stdout", "stderr", "output")) if isinstance(resp, dict) else str(resp or "")
        text = s.after(command, output if command else "", read=read, read_range=rng,
                       authored=path if tool in EDIT_TOOLS and not failed else None,
                       wrote=path if tool in WRITE_TOOLS and not failed else None, cwd=cwd)
        return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}} if text else None


def main():
    t0 = time.perf_counter()
    try:
        reply = handle(json.loads(sys.stdin.buffer.read()))  # UTF-8 whatever the locale (Windows)
    except Exception as e:  # noqa: BLE001  never break the session: log and stay silent
        log = os.path.join(default_state_dir(), "errors.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {type(e).__name__}: {e}\n")
        return 0
    if reply:
        json.dump(reply, sys.stdout)
    if os.environ.get("ALIGN_TIMING"):
        sys.stderr.write(f"ALIGN hook {1000 * (time.perf_counter() - t0):.1f} ms\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
