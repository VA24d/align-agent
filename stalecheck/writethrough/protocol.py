"""A JSON command for any agent whose hooks can run a program: `stalecheck-wt`.

Reads one event as JSON on stdin and writes one JSON reply on stdout:

    {"event": "start",    "root": ".", "session": "s1", "fresh": true}
    {"event": "read",     "root": ".", "session": "s1", "path": "app.py", "lines": [10, 40]}
    {"event": "authored", "root": ".", "session": "s1", "path": "app.py"}
    {"event": "before",   "root": ".", "session": "s1", "command": "sed -i '60d' app.py"}
    {"event": "after",    "root": ".", "session": "s1", "command": "...", "output": "...",
                          "read": "app.py", "lines": [1, 40], "authored": "b.py"}   (read, lines, authored optional)

    -> {"context": "<text to append to the tool result>"}   (after; omitted when empty)
    -> {"deny": "<reason to show the model>"}                (before; omitted when allowed)

`--deny-exit 2` makes a refusal exit with that status and the reason on stderr, which is how
several hook systems block a tool. `--text` writes the context as plain text instead of JSON.
"""

import argparse
import json
import os
import sys

from .engine import Session


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
            s.authored(ev["path"])
            return {}
        if kind == "before":
            reason = s.before(ev.get("command") or "")
            return {"deny": reason} if reason else {}
        if kind == "after":
            text = s.after(ev.get("command") or "", ev.get("output") or "", read=ev.get("read"),
                           read_range=ev.get("lines"), authored=ev.get("authored"))
            return {"context": text} if text else {}
        raise ValueError(f"unknown event {kind!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stalecheck-wt", description=__doc__.split("\n\n")[0])
    ap.add_argument("--deny-exit", type=int, default=0, help="exit status for a refusal (reason on stderr)")
    ap.add_argument("--text", action="store_true", help="write the context as plain text")
    args = ap.parse_args(argv)
    reply = respond(json.load(sys.stdin))
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
