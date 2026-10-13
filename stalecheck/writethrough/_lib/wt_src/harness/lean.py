"""Which written files need no write-through block.

A block costs tokens in every later prompt, so it should appear only when it
tells the agent something it cannot already know. Three cases do not:

* a whole-file write whose content the agent supplied (a heredoc or echo into
  the file): it typed every byte;
* a file the same command printed after its last write to it: the output
  already shows the result;
* the submission's own patch.txt.

The rules match the replay's (src/metrics/replay.py): an overwrite leaves
known content and a print is a view, so skipping the block never turns a
fresh edit into a stale one in the score.
"""

from typing import Dict, Iterable, Optional, Set

from wt_src.metrics.replay import _live_events

SUBMISSION_FILES = {"patch.txt"}


def redundant_writes(command: str, observation: str, writes: Iterable[str],
                     scripts: Optional[Dict[str, str]] = None) -> Set[str]:
    """The written files whose block would repeat what the agent already has."""
    writes = list(writes)
    events = _live_events({"command": command, "observation": observation, "writes_detected": writes},
                          scripts if scripts is not None else {})
    out: Set[str] = set()
    for w in writes:
        if w.rsplit("/", 1)[-1] in SUBMISSION_FILES:
            out.add(w)
            continue
        mine = [(op, kind) for op, path, kind, landed in events if path == w]
        kinds = [kind for op, kind in mine if op == "W"]
        if kinds and all(k == "overwrite" for k in kinds):
            out.add(w)
        elif mine and mine[-1][0] == "R" and any(op == "W" for op, _ in mine):
            out.add(w)  # printed after its last write
    return out
