"""Which written files need no write-through block.

A block costs tokens in every later prompt, so it should appear only when it
tells the agent something it cannot already know. Three cases do not:

* a whole-file write whose content the agent supplied (a literal heredoc, echo
  or printf into the file, with nothing expanded and nothing else in the
  command able to change the file after it): it typed every byte;
* a file the same command printed whole (cat, bat, nl -ba; not piped on, not
  redirected) after its last write to it: the output already shows the result;
* the submission's own patch.txt.

The rules match the replay's (src/metrics/replay.py): an overwrite leaves
known content and a print is a view, so skipping the block never turns a
fresh edit into a stale one in the score.
"""

from typing import Dict, Iterable, Optional, Set

import os
import re

from src.metrics.replay import _live_events
from src.parsers.file_events import _strip_prefixes, _units_with_cwd, command_units

SUBMISSION_FILES = {"patch.txt"}


# commands that cannot change a file they do not redirect into
READ_ONLY = {"cd", "pushd", "popd", "ls", "cat", "bat", "batcat", "nl", "head", "tail", "grep", "rg", "wc",
             "echo", "printf", "pwd", "true", "test", "[", "stat", "file", "mkdir", "diff"}
WHOLE_PRINTERS = {"cat", "bat", "batcat", "nl"}


def _placed(command: str):
    """(argv, unit, directory, piped on) for each simple command; directory None when unknown."""
    units = command_units(command)
    placed, _ = _units_with_cwd(units, "")
    out = []
    for k, (u, d) in enumerate(placed):
        argv = _strip_prefixes(u["argv"])
        piped = any(v["pipe"] == u["pipe"] and v.get("depth", 0) == u.get("depth", 0) for v in units[k + 1:])
        out.append((argv, u, d, piped))
    return out


def _names(argv, d) -> Set[str]:
    return {os.path.normpath(os.path.join(d, a)) for a in argv[1:] if not a.startswith("-")} if d is not None else set()


def _literal_writer(argv, u, d, w, quoted_heredoc: bool) -> bool:
    """A unit that writes `w` with text the agent typed: echo/printf/cat-heredoc redirected into it."""
    if not argv or os.path.basename(argv[0]) not in ("echo", "printf", "cat") or d is None:
        return False
    if not any(op in (">", ">|") and os.path.normpath(os.path.join(d, tgt)) == w for op, tgt in u["redirs"]):
        return False
    if os.path.basename(argv[0]) == "cat" and (len(argv) > 1 or u["heredoc"] is None):
        return False
    body = u["heredoc"] if u["heredoc"] is not None else " ".join(argv[1:])
    return not (("$" in body or "`" in body) and not (u["heredoc"] is not None and quoted_heredoc))


def redundant_writes(command: str, observation: str, writes: Iterable[str],
                     scripts: Optional[Dict[str, str]] = None) -> Set[str]:
    """The written files whose block would repeat what the agent already has."""
    writes = list(writes)
    events = _live_events({"command": command, "observation": observation, "writes_detected": writes},
                          scripts if scripts is not None else {})
    placed = _placed(command)
    quoted = bool(re.search(r"""<<-?\s*['"]""", command))
    out: Set[str] = set()
    for w in writes:
        if w.rsplit("/", 1)[-1] in SUBMISSION_FILES:
            out.add(w)
            continue
        mine = [(op, kind) for op, path, kind, landed in events if path == w]
        kinds = [kind for op, kind in mine if op == "W"]
        writers = [k for k, (argv, u, d, _) in enumerate(placed)
                   if any(op in (">", ">>", ">|") and d is not None and os.path.normpath(os.path.join(d, t)) == w
                          for op, t in u["redirs"])]
        others = [k for k in range(len(placed)) if k not in writers]
        if (kinds and all(k == "overwrite" for k in kinds) and writers
                and all(_literal_writer(*placed[k][:3], w, quoted) and not any(
                    v["pipe"] == placed[k][1]["pipe"] for v in (placed[j][1] for j in others)) for k in writers)
                and all(placed[k][0] and os.path.basename(placed[k][0][0]) in READ_ONLY for k in others)):
            out.add(w)
            continue
        last_w = max(writers) if writers else -1
        if mine and mine[-1][0] == "R" and any(op == "W" for op, _ in mine) and any(
                k > last_w and argv and os.path.basename(argv[0]) in WHOLE_PRINTERS and not piped
                and not u["redirs"] and w in _names(argv, d)
                and (os.path.basename(argv[0]) != "nl" or "-ba" in argv or ("-b" in argv and "a" in argv))
                for k, (argv, u, d, piped) in enumerate(placed)):
            out.add(w)  # printed whole after its last write
    return out
