"""Stale verification: the last test run happened before the last edit.

An agent that runs its tests, sees them pass, edits again and submits holds a
belief about the code that is older than the code: the same stale-read shape as
a stale file view, one level up, on a result instead of a file.

A run's state is one of
  no_code_write   it never changed a counted file
  never_tested    it changed code and never ran a test command
  fresh           the last test run came after the last counted write
  stale           a counted write came after the last test run
Counted writes are landed writes to source files that are not tests; the
caller can narrow them further (for example, to the files in the submitted
patch) with `keep`.
"""

import os
import re
from typing import Callable, Dict, List, Optional, Sequence

from src.parsers.file_events import command_units

_RUNNERS = {"pytest", "py.test", "tox", "nox", "jest", "mocha", "vitest", "ctest", "rspec", "phpunit", "karma", "ava"}
_PAIRS = {("go", "test"), ("cargo", "test"), ("npm", "test"), ("npm", "t"), ("yarn", "test"), ("pnpm", "test"),
          ("make", "test"), ("make", "check"), ("mvn", "test"), ("mvn", "verify"), ("dotnet", "test"), ("swift", "test"),
          ("bun", "test"), ("deno", "test"), ("mix", "test"), ("sbt", "test")}
_MODULES = {"pytest", "unittest", "nose2", "nose"}
TEST_FILE = re.compile(r"(^|/)(tests?|__tests__|spec|testing)/|(^|/)test_[^/]*\.py$|_test\.(py|go)$|\.(test|spec)\.[cm]?[jt]sx?$"
                       r"|(^|/)[^/]*Test\.(java|kt|ts|tsx|js)$|_spec\.rb$")
CODE = {".py", ".go", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".java",
        ".kt", ".rb", ".php", ".scala", ".swift", ".cs", ".ml", ".mli", ".ex", ".exs", ".lua", ".vue", ".svelte", ".sh"}


def _is_test_argv(argv: List[str]) -> bool:
    words = [os.path.basename(a) for a in argv if not re.match(r"^\w+=", a)]  # skip VAR=value prefixes
    if not words:
        return False
    head = words[0]
    if head in ("npx", "bunx", "exec") or (head == "bundle" and len(words) > 1 and words[1] == "exec"):
        words = words[2:] if head == "bundle" else words[1:]
        if not words:
            return False
        head = words[0]
    if head in ("timeout", "time", "nice", "env"):
        rest = [w for w in words[1:] if not re.match(r"^(-|\d)", w)]
        return _is_test_argv(rest)
    if head in _RUNNERS:
        return True
    if re.match(r"python[\d.]*$", head):
        return len(words) > 2 and words[1] == "-m" and words[2] in _MODULES
    if head in ("npm", "yarn", "pnpm") and len(words) > 2 and words[1] == "run":
        return words[2].startswith("test")
    if re.match(r"(gradle|gradlew)$", head):
        return any(w in ("test", "check") for w in words[1:])
    if head == "mvn":
        return any(w in ("test", "verify") for w in words[1:])
    return len(words) > 1 and (head, words[1]) in _PAIRS


def is_test_command(cmd: str) -> bool:
    """Whether any simple command in `cmd` runs a test suite."""
    return any(_is_test_argv(u["argv"]) for u in command_units(cmd or ""))


def counted(path: str) -> bool:
    """A source file that is not a test."""
    return os.path.splitext(path)[1] in CODE and not TEST_FILE.search(path)


def _last_positions(cmd: str, writes: Sequence[str]):
    """(index of the last test unit, index of the last unit that names a written file) in one command."""
    units = command_units(cmd or "")
    last_test, last_write = None, None
    names = {os.path.basename(w) for w in writes}
    for i, u in enumerate(units):
        if _is_test_argv(u["argv"]):
            last_test = i
        text = " ".join(u["argv"]) + " " + " ".join(str(r) for r in u.get("redirs") or []) + " " + str(u.get("heredoc") or "")
        if any(n and n in text for n in names):
            last_write = i
    return last_test, last_write


def verification_state(steps: Sequence[Dict], keep: Optional[Callable[[str], bool]] = None) -> Dict:
    """Each step: {"cmd": command text, "writes": landed paths}. Returns the
    state, the step of the last test run, and the files written after it."""
    keep = keep or (lambda p: True)
    last_test_at = None  # (step, unit) of the last test run
    writes_at = {}  # path -> (step, unit) of its last counted write
    for si, s in enumerate(steps):
        writes = [w for w in s.get("writes") or [] if counted(w) and keep(w)]
        test_i, write_i = _last_positions(s.get("cmd") or "", writes)
        if test_i is not None:
            last_test_at = (si, test_i)
        for w in writes:
            # A write the command text does not name (an editor call, a
            # script) is placed at the start of its step's commands.
            writes_at[w] = (si, write_i if write_i is not None else -1)
    if not writes_at:
        return {"state": "no_code_write", "last_test": last_test_at, "stale_files": []}
    if last_test_at is None:
        return {"state": "never_tested", "last_test": None, "stale_files": []}
    stale = sorted(w for w, at in writes_at.items() if at > last_test_at)
    return {"state": "stale" if stale else "fresh", "last_test": last_test_at, "stale_files": stale}
