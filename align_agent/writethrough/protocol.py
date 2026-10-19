"""A JSON command for any agent whose hooks can run a program: `align-agent-wt`.

Reads one event as JSON on stdin and writes one JSON reply on stdout:

    {"event": "start",    "root": ".", "session": "s1", "fresh": true}
    {"event": "read",     "root": ".", "session": "s1", "path": "app.py", "lines": [10, 40]}
    {"event": "authored", "root": ".", "session": "s1", "path": "app.py", "whole": false}
    {"event": "before",   "root": ".", "session": "s1", "command": "sed -i '60d' app.py", "cwd": "src"}
    {"event": "after",    "root": ".", "session": "s1", "command": "...", "output": "...", "cwd": "src",
                          "read": "app.py", "lines": [1, 40], "authored": "b.py"}   (cwd, read, lines, authored optional)

    -> {"context": "<text to append to the tool result>"}   (after; omitted when empty)
    -> {"deny": "<reason to show the model>"}                (before; omitted when allowed)

`--deny-exit 2` makes a refusal exit with that status and the reason on stderr, which is how
several hook systems block a tool. `--text` writes the context as plain text instead of JSON.
An error is logged to <state_dir>/errors.log and the reply is empty: the agent is never blocked.
"""

import argparse
import json
import os
import sys
import time

from .engine import Session, default_state_dir


def respond(ev):
    root = ev.get("root") or os.getcwd()
    with Session(root, ev.get("session") or "default", ev.get("state_dir")) as s:
        kind = ev.get("event")
        if kind == "start":
            s.start(fresh=ev.get("fresh", True))
            return {}
        if kind == "read":
            lines = ev.get("lines") or (None, None)
            s.read(ev["path"], *lines)
            return {}
        if kind == "authored":
            s.authored(ev["path"], whole=bool(ev.get("whole")))
            return {}
        if kind == "before":
            reason = s.before(ev.get("command") or "", cwd=ev.get("cwd"))
            return {"deny": reason} if reason else {}
        if kind == "after":
            text = s.after(ev.get("command") or "", ev.get("output") or "", read=ev.get("read"),
                           read_range=ev.get("lines"), authored=ev.get("authored"), cwd=ev.get("cwd"))
            return {"context": text} if text else {}
        raise ValueError(f"unknown event {kind!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="align-agent-wt", description=__doc__.split("\n\n")[0])
    ap.add_argument("--deny-exit", type=int, default=0, help="exit status for a refusal (reason on stderr)")
    ap.add_argument("--text", action="store_true", help="write the context as plain text")
    args = ap.parse_args(argv)
    try:
        reply = respond(json.loads(sys.stdin.buffer.read()))  # UTF-8 whatever the locale (Windows)
    except Exception as e:  # noqa: BLE001  fail open: a hook that errors must not block the tool
        log = os.path.join(default_state_dir(), "errors.log")
        try:
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} align-agent-wt {type(e).__name__}: {e}\n")
        except OSError:
            pass
        reply = {}
    if "deny" in reply and args.deny_exit:
        sys.stderr.write(reply["deny"] + "\n")
        return args.deny_exit
    if args.text:
        sys.stdout.write(reply.get("context") or reply.get("deny") or "")
    elif reply:
        json.dump(reply, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
