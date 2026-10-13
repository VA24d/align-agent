"""The tail-diff coherence mechanism, decoupled from any one agent scaffold.

Unlike full-file JIT Tail Refresh (Arm 7), which dumps the entire file (2k-4.2k
lines, $O(N)$ tokens) and rewrites earlier turns in the history (annihilating
the KV-cache prefix), this mechanism:
1. Is strictly append-only: past conversation turns are never modified,
   preserving 100% of the prefix KV-cache in local engines (vLLM, SGLang) and APIs.
2. Computes the net unified diff or line-numbered diff hunk between what the
   agent last saw (held view) and the current live file on disk ($O(\\Delta)$ tokens).
3. Naturally captures multiple disjoint hunks across large files (solving the
   single-window limitation of Arm 9 Write-Through Viewport, which misses edits
   outside a +/- 25 line window).
4. Handles sequential conflicting overwrites by collapsing intermediate mutations
   into the exact net diff from the agent's held view to live disk.
"""

import copy
import difflib
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple

TAIL_DIFF_CONTEXT = 3
TAIL_DIFF_MAX_FILES = 3
TAIL_DIFF_MAX_LINE = 200
TAIL_DIFF_MAX_CHARS = 10_000


class _BindingsSource(Protocol):
    bindings: List[object]  # each binding needs .file_path and .file_content


def _is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return True


def _clip(line: str) -> str:
    body = line.rstrip("\n")
    if len(body) <= TAIL_DIFF_MAX_LINE:
        return line
    tail = "\n" if line.endswith("\n") else ""
    return f"{body[:TAIL_DIFF_MAX_LINE]} ...[{len(body) - TAIL_DIFF_MAX_LINE} more characters]{tail}"


def format_line_numbered_diff(
    held_content: str,
    disk_content: str,
    filename: str,
    context: int = TAIL_DIFF_CONTEXT,
    step_seen: Optional[int] = None,
    current_step: Optional[int] = None,
) -> str:
    """Format a diff between held content and live disk with explicit live line numbers."""
    held_lines = held_content.splitlines()
    disk_lines = disk_content.splitlines()
    matcher = difflib.SequenceMatcher(None, held_lines, disk_lines)

    step_info = f" (held at Step {step_seen})" if step_seen is not None else ""
    curr_info = f" (Step {current_step})" if current_step is not None else ""
    
    blocks: List[str] = []
    for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
        if tag == "equal":
            continue

        c_alo = max(0, alo - context)
        c_ahi = min(len(held_lines), ahi + context)
        c_blo = max(0, blo - context)
        c_bhi = min(len(disk_lines), bhi + context)

        hunk_header = (
            f"@@ Held Lines {c_alo + 1}-{c_ahi}{step_info} "
            f"-> Live Disk Lines {c_blo + 1}-{c_bhi}{curr_info} @@"
        )
        lines_out = [hunk_header]

        # Context lines before change (using current live disk line numbers)
        for lno in range(c_blo, blo):
            lines_out.append(f"{lno + 1:4d}   {_clip(disk_lines[lno])}")

        # Removed lines from held view
        for lno in range(alo, ahi):
            lines_out.append(f"     - {_clip(held_lines[lno])}")

        # Added / modified lines on disk with current line numbers
        for lno in range(blo, bhi):
            lines_out.append(f"{lno + 1:4d} + {_clip(disk_lines[lno])}")

        # Context lines after change
        for lno in range(bhi, c_bhi):
            lines_out.append(f"{lno + 1:4d}   {_clip(disk_lines[lno])}")

        blocks.append("\n".join(lines_out))

    if not blocks:
        return ""

    return (
        f"[TAIL DIFF: File '{filename}' mutated on disk vs your held view]:\n"
        + "\n\n".join(blocks)
    )


def tail_diff_blocks(
    workspace_dir: str,
    files: List[str],
    freshness_hook: Optional[_BindingsSource] = None,
    before_contents: Optional[Dict[str, str]] = None,
    context_lines: int = TAIL_DIFF_CONTEXT,
    line_numbered: bool = True,
    current_step: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """Compute tail-diff blocks for the specified files against held views.

    The held view is the latest binding from a step before `current_step`: a
    read in the step that wrote the file was registered with the file as it
    stood after the write, and diffing against it would show nothing. With no
    earlier view, the content before this step is the baseline.

    Returns the formatted diff text and the list of files successfully shown.
    """
    if not files:
        return "", []

    bindings = getattr(freshness_hook, "bindings", None) or []
    from .viewport import _ignored  # generated output is not file state (harness v12)
    ignored = _ignored(workspace_dir, list(files))
    files = [f for f in files if f not in ignored]
    known = set(before_contents or ()) | {getattr(b, "file_path", None) for b in bindings}
    blocks: List[str] = []
    shown: List[str] = []
    used_chars = 0

    for rel_path in sorted(files, key=lambda p: (p not in known, p)):
        abs_path = os.path.join(workspace_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
        if not os.path.isfile(abs_path) or _is_binary(abs_path):
            continue
        if len(shown) >= TAIL_DIFF_MAX_FILES:
            break

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                disk_content = f.read()
        except OSError:
            continue

        # Locate held content and step ID
        held_content = None
        step_seen = None
        for b in reversed(bindings):
            if getattr(b, "file_path", None) != rel_path:
                continue
            if current_step is not None and getattr(b, "step_id", None) is not None and b.step_id >= current_step:
                continue  # this step's own read: the file after the write
            held_content = getattr(b, "file_content", None) or getattr(b, "raw_content", None)
            step_seen = getattr(b, "step_id", None)
            break

        if held_content is None and before_contents and rel_path in before_contents:
            held_content = before_contents[rel_path]

        if held_content is None:
            # No prior view exists: agent never read this file
            continue

        if held_content == disk_content:
            # File matches held view: no staleness
            continue

        if line_numbered:
            diff_text = format_line_numbered_diff(
                held_content, disk_content, rel_path, context=context_lines, step_seen=step_seen
            )
        else:
            diff_lines = list(difflib.unified_diff(
                held_content.splitlines(keepends=True),
                disk_content.splitlines(keepends=True),
                fromfile=f"{rel_path} (held at Step {step_seen})",
                tofile=f"{rel_path} (current on disk)",
                n=context_lines,
            ))
            diff_text = f"[TAIL DIFF: File '{rel_path}']\n" + "".join(_clip(l) for l in diff_lines)

        if not diff_text:
            continue

        if used_chars + len(diff_text) > TAIL_DIFF_MAX_CHARS:
            diff_text = diff_text[: TAIL_DIFF_MAX_CHARS - used_chars] + "\n...[truncated]"

        blocks.append(diff_text)
        shown.append(rel_path)
        used_chars += len(diff_text)
        if used_chars >= TAIL_DIFF_MAX_CHARS:
            break

    if not blocks:
        return "", []

    return "\n\n" + "\n\n".join(blocks), shown


def apply_tail_diff(
    messages: List[Dict[str, Any]],
    bindings: List[object],
    workspace_root: str = ".",
    line_numbered: bool = True,
) -> List[Dict[str, Any]]:
    """Transform context by appending net diffs of stale files to prompt tail.

    Crucially:
    - messages[:-1] is NEVER touched, guaranteeing 100% prefix KV-cache reuse.
    - Only files that are currently stale are diffed against the agent's held view.
    - The diff is appended to the tail message.
    """
    transformed = copy.deepcopy(messages)
    if not transformed:
        return transformed

    # Find distinct stale files
    latest_by_file: Dict[str, object] = {}
    for b in bindings:
        if getattr(b, "is_invalid", False):
            latest_by_file[getattr(b, "file_path", "")] = b

    if not latest_by_file:
        return transformed

    diff_blocks: List[str] = []
    for f_path, binding in latest_by_file.items():
        abs_p = os.path.join(workspace_root, f_path) if not os.path.isabs(f_path) else f_path
        if os.path.exists(abs_p) and os.path.isfile(abs_p):
            try:
                with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                    disk_content = f.read()
                held_content = getattr(binding, "file_content", None) or getattr(binding, "raw_content", None)
                if held_content is None or held_content == disk_content:
                    continue

                step_seen = getattr(binding, "step_id", None)
                inval_step = getattr(binding, "invalidation_step", None)

                if line_numbered:
                    diff_block = format_line_numbered_diff(
                        held_content=held_content,
                        disk_content=disk_content,
                        filename=f_path,
                        context=TAIL_DIFF_CONTEXT,
                        step_seen=step_seen,
                        current_step=inval_step,
                    )
                else:
                    diff_lines = list(difflib.unified_diff(
                        held_content.splitlines(keepends=True),
                        disk_content.splitlines(keepends=True),
                        fromfile=f"{f_path} (held at Step {step_seen})",
                        tofile=f"{f_path} (current on disk at Step {inval_step})",
                        n=TAIL_DIFF_CONTEXT,
                    ))
                    diff_block = f"[TAIL DIFF: File '{f_path}']\n" + "".join(_clip(l) for l in diff_lines)

                if diff_block:
                    diff_blocks.append(diff_block)
            except Exception:
                pass

    if diff_blocks:
        forwarding_text = "\n\n" + "\n\n".join(diff_blocks)
        if transformed[-1].get("role") in ["user", "tool"]:
            transformed[-1]["content"] += forwarding_text
        else:
            transformed.append({"role": "user", "content": forwarding_text.strip()})

    return transformed
