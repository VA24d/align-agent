"""A syntax check for the files a write changed: stdlib only, and nothing in the project is run.

`broke(path, old, new)` returns "line N: message" when `new` no longer parses but `old` did, else None.
Python is compiled (never executed), JSON and TOML are loaded, Go goes through `gofmt -e` when it is
installed. Other languages are skipped. A file that did not parse before the write is not reported again.
Set ALIGN_SYNTAX=0 to turn the check off.
"""

import json
import os
import re
import shutil
import subprocess
import sys

GOFMT_TIMEOUT = 3  # seconds
MAX_BYTES = 400_000


def _python(path, text):
    try:
        compile(text, path, "exec", dont_inherit=True)
    except SyntaxError as e:
        msg = e.msg or "invalid syntax"
        if sys.version_info < (3, 12):  # newer syntax (match, type aliases, PEP 695) fails on an old interpreter
            msg += f" (checked with Python {sys.version_info[0]}.{sys.version_info[1]}; newer syntax may be valid)"
        return e.lineno or 0, msg
    except (ValueError, RecursionError, MemoryError) as e:  # null bytes, absurd nesting
        return 0, str(e)[:80]
    return None


def _json(path, text):
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return e.lineno, e.msg
    except RecursionError:
        return 0, "nesting too deep"
    return None


def _toml(path, text):
    try:
        import tomllib
    except ImportError:  # Python 3.10
        return None
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        m = re.search(r"\(at line (\d+)", str(e))
        return (int(m.group(1)) if m else 0), re.sub(r"\s*\(at line.*", "", str(e))[:80]
    return None


def _go(path, text):
    exe = shutil.which("gofmt")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "-e"], input=text.encode("utf-8", "surrogateescape"), capture_output=True, timeout=GOFMT_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode == 0:
        return None
    m = re.search(r"<standard input>:(\d+):\d+: (.*)", r.stderr.decode("utf-8", "replace"))
    return (int(m.group(1)), m.group(2)[:80]) if m else None


CHECKS = {".py": _python, ".json": _json, ".toml": _toml, ".go": _go}


def _check(path, text):
    fn = CHECKS.get(os.path.splitext(path)[1].lower())
    if fn is None or text is None or len(text) > MAX_BYTES:
        return None
    return fn(path, text)


def broke(path, old, new):
    """'line N: message' if `new` fails to parse and `old` parsed; None if fine, unknown or already broken."""
    if os.environ.get("ALIGN_SYNTAX") == "0":
        return None
    bad = _check(path, new)
    if bad is None:
        return None
    if old is not None and _check(path, old) is not None:
        return None
    return f"line {bad[0]}: {bad[1]}" if bad[0] else bad[1]
