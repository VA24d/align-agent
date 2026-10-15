"""Intervention arms for RQ2: what each one does to the prompt, and its names.

Three families (names and one-line effects are in ARM_INFO below, the single
source every script, table and figure reads):

- Baselines: Control, Re-read reminder, Recency eviction, and Volume-matched
  eviction (the control for Stale-view removal).
- Retroactive rewrite: Stale flag, Stale-view removal, Changed-hunk removal,
  In-place refresh, Tail refresh. All rewrite earlier turns, so all break the
  prefix cache.
- Tool boundary: Write-through viewport, which never rewrites an earlier turn.

Enum values are the on-disk keys in traces and results and never change.
"""

import os
import copy
import re
import difflib
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from .freshness_hook import ObservationBinding


class InterventionArm(Enum):
    ARM_0_PROMPT_ONLY = "prompt_only_baseline"
    ARM_1_CONTROL = "control"
    ARM_2_POSITIONAL = "positional_elision"
    ARM_3_MARKED = "freshness_marked"
    ARM_4_MASKED = "freshness_masked"
    ARM_5_VOLUME_CONTROL = "volume_matched_control"
    ARM_6_IN_PLACE = "in_place_replacement"
    ARM_7_JIT_PREFETCH = "jit_prefetch"
    ARM_8_FINE_GRAINED = "fine_grained_invalidation"
    ARM_9_WRITE_THROUGH = "write_through_viewport"
    ARM_10_WRITE_THROUGH_DIFF = "write_through_diff"
    ARM_11_SYNCED_CONTEXT = "synced_context"
    ARM_12_DEPENDENCY_WRITE_THROUGH = "dependency_write_through"
    ARM_13_WRITE_THROUGH_DIFF_LEAN = "write_through_diff_lean"
    ARM_14_WRITE_THROUGH_DIFF_LINES = "write_through_diff_lines"


@dataclass(frozen=True)
class ArmInfo:
    """How an arm is named and grouped everywhere outside the enum.

    The enum value is the on-disk key (trace filenames, results records) and
    never changes. Every script, table and figure takes its labels from here,
    so an arm has exactly one name per medium.
    """

    family: str            # "baseline" | "retroactive" | "tool_boundary"
    display: str           # table / paper name
    short: str             # figure axis label
    cli: tuple             # accepted --arm spellings; first is canonical
    macro: str             # suffix of the generated paper macros (stable)
    effect: str            # what it does to the prompt, one sentence
    core: bool = False     # reported in the main paper; the rest in the extended report


FAMILIES = {
    "baseline": "Baselines",
    "retroactive": "Retroactive rewrite",
    "tool_boundary": "Tool boundary",
    "sync": "Synchronized context",
}

ARM_INFO = {
    InterventionArm.ARM_1_CONTROL: ArmInfo(
        "baseline", "Control", "Control", ("control",), "Control",
        "History is left unchanged.", core=True),
    InterventionArm.ARM_0_PROMPT_ONLY: ArmInfo(
        "baseline", "Re-read reminder", "Reminder", ("prompt_only", "prompt_only_baseline", "reminder"),
        "PromptOnly",
        "Adds an instruction to re-read a file before editing it to the system prompt; history unchanged.",
        core=True),
    InterventionArm.ARM_2_POSITIONAL: ArmInfo(
        "baseline", "Recency eviction", "Recency", ("positional", "positional_elision", "recency"),
        "Positional",
        "Above 32K estimated tokens, replaces the oldest observations with a placeholder, whether stale or not."),
    InterventionArm.ARM_5_VOLUME_CONTROL: ArmInfo(
        "baseline", "Volume-matched eviction", "Vol.-matched",
        ("volume_control", "volume_matched_control"), "VolumeMatched",
        "Evicts as many tokens as stale-view removal would, taken oldest-first from observations "
        "that are not stale; the control for stale-view removal."),
    InterventionArm.ARM_3_MARKED: ArmInfo(
        "retroactive", "Stale flag", "Flag", ("marked", "freshness_marked", "stale_flag"), "Marked",
        "Appends a notice after each stale observation; the stale text stays in place."),
    InterventionArm.ARM_4_MASKED: ArmInfo(
        "retroactive", "Stale-view removal", "Removal", ("masked", "freshness_masked", "removal"), "Masked",
        "Replaces each stale observation with a placeholder telling the agent to re-read the file.",
        core=True),
    InterventionArm.ARM_8_FINE_GRAINED: ArmInfo(
        "retroactive", "Changed-hunk removal", "Hunk removal",
        ("fine_grained", "fine_grained_invalidation", "hunk_removal"), "FineGrained",
        "Removes only the changed hunks of each stale observation and keeps the unchanged lines."),
    InterventionArm.ARM_6_IN_PLACE: ArmInfo(
        "retroactive", "In-place refresh", "In-place", ("in_place", "in_place_replacement"), "InPlace",
        "Overwrites each stale observation, where it sits in the history, with the file's full current content.",
        core=True),
    InterventionArm.ARM_7_JIT_PREFETCH: ArmInfo(
        "retroactive", "Tail refresh", "Tail refresh", ("jit_prefetch", "tail_refresh"), "JIT",
        "Replaces each stale observation with a placeholder and appends the file's full current "
        "content at the end of the prompt."),
    InterventionArm.ARM_9_WRITE_THROUGH: ArmInfo(
        "tool_boundary", "Write-through viewport", "Write-through",
        ("write_through_viewport", "write_through"), "WriteThrough",
        "After a step writes a file, appends a line-numbered window around the edit to that step's own "
        "tool output; earlier turns are never rewritten.", core=True),
    InterventionArm.ARM_10_WRITE_THROUGH_DIFF: ArmInfo(
        "tool_boundary", "Write-through diff", "WT diff",
        ("write_through_diff", "wt_diff"), "WriteThroughDiff",
        "After a step writes a file, appends the net line-numbered diff from the version the agent last "
        "held to the file on disk, every hunk; earlier turns are never rewritten."),
    InterventionArm.ARM_11_SYNCED_CONTEXT: ArmInfo(
        "sync", "Synced context", "Synced",
        ("synced_context", "synced", "corvus"), "SyncedContext",
        "A CORVUS-style baseline: every prompt ends with the current contents of each file the agent has "
        "viewed, read from disk, and observations that were only a read of such a file are elided."),
    InterventionArm.ARM_12_DEPENDENCY_WRITE_THROUGH: ArmInfo(
        "tool_boundary", "Dependency write-through", "Dep. WT",
        ("dependency_write_through", "dep_wt"), "DependencyWriteThrough",
        "Write-through diff, plus a notice listing uses in other files of names the write removed or "
        "whose parameters it changed; earlier turns are never rewritten."),
    InterventionArm.ARM_13_WRITE_THROUGH_DIFF_LEAN: ArmInfo(
        "tool_boundary", "Write-through diff (lean)", "WT diff lean",
        ("write_through_diff_lean", "wt_diff_lean"), "WriteThroughDiffLean",
        "Write-through diff, without a block for writes that tell the agent nothing new: whole files it "
        "typed itself, files the same command printed after writing, and patch.txt."),
    InterventionArm.ARM_14_WRITE_THROUGH_DIFF_LINES: ArmInfo(
        "tool_boundary", "Write-through diff (lean) with line numbers", "WT diff lines",
        ("write_through_diff_lines", "wt_diff_lines"), "WriteThroughDiffLines",
        "The lean write-through diff, plus a note of every stretch of the file that moved since the agent "
        "last read it, with its offset, as the shipped hook does (tail_diff.line_number_note)."),
}

# Table and figure order: family by family, as listed above.
ARM_ORDER = sorted(ARM_INFO, key=lambda a: (list(FAMILIES).index(ARM_INFO[a].family),
                                            list(ARM_INFO).index(a)))


def arm_from_cli(name: str) -> "InterventionArm":
    """The arm a --arm spelling names (any alias, or the enum value)."""
    for arm, info in ARM_INFO.items():
        if name == arm.value or name in info.cli:
            return arm
    raise ValueError(f"unknown arm {name!r}")


def arm_info(value: str) -> ArmInfo:
    """Info for an on-disk arm value, as stored in traces and results."""
    return ARM_INFO[InterventionArm(value)]


# Appended to the system prompt for ARM_0_PROMPT_ONLY. One copy, shared by
# every scaffold, so the prompt baseline cannot drift between harnesses.
PROMPT_ONLY_POLICY = """

[POLICY: PROMPT-ONLY RE-READ INSTRUCTION]
Before modifying or referencing any file, if you have not inspected its content recently \
or if it was modified previously, always re-read the latest on-disk state with `cat <file>`. \
Never make assumptions about line numbers or file contents based on past observations."""


@dataclass(frozen=True)
class ArmRemediation:
    """What an arm does to a stale observation before the prompt is sent.

    Metrics need this to tell "the hazard never happened" from "the hazard
    happened and the arm repaired it". Scoring every arm on invalidation
    counts alone charges the repairing arms for the hazards they fix, which
    ranked the arms that merely edited least at the top.

    withdraws: the stale text is replaced, so the model cannot read it.
    annotates: the stale text is kept and a warning is attached.
    forwards:  current file content is appended at the prompt tail.
    refreshes: current file content is written over the stale text, where
               it sat in the history.
    partial:   only the changed region is withdrawn; untouched lines remain,
               and those lines are not stale.
    """

    withdraws: bool = False
    annotates: bool = False
    forwards: bool = False
    refreshes: bool = False
    partial: bool = False

    @property
    def supplies_current(self) -> bool:
        """The model can see the file's current content without re-reading it."""
        return self.forwards or self.refreshes


# Declared next to the transforms above so the two cannot drift apart. Arm 2
# evicts by recency rather than by staleness, so it withdraws stale content
# only by coincidence and is not counted as remediating.
ARM_REMEDIATION = {
    InterventionArm.ARM_0_PROMPT_ONLY: ArmRemediation(),
    InterventionArm.ARM_1_CONTROL: ArmRemediation(),
    InterventionArm.ARM_2_POSITIONAL: ArmRemediation(),
    InterventionArm.ARM_3_MARKED: ArmRemediation(annotates=True),
    InterventionArm.ARM_4_MASKED: ArmRemediation(withdraws=True),
    InterventionArm.ARM_5_VOLUME_CONTROL: ArmRemediation(),
    InterventionArm.ARM_6_IN_PLACE: ArmRemediation(withdraws=True, refreshes=True),
    InterventionArm.ARM_7_JIT_PREFETCH: ArmRemediation(withdraws=True, forwards=True),
    InterventionArm.ARM_8_FINE_GRAINED: ArmRemediation(withdraws=True, partial=True),
    InterventionArm.ARM_9_WRITE_THROUGH: ArmRemediation(forwards=True),
    InterventionArm.ARM_10_WRITE_THROUGH_DIFF: ArmRemediation(forwards=True),
    InterventionArm.ARM_11_SYNCED_CONTEXT: ArmRemediation(forwards=True),
    InterventionArm.ARM_12_DEPENDENCY_WRITE_THROUGH: ArmRemediation(forwards=True),
    InterventionArm.ARM_13_WRITE_THROUGH_DIFF_LEAN: ArmRemediation(forwards=True),
    InterventionArm.ARM_14_WRITE_THROUGH_DIFF_LINES: ArmRemediation(forwards=True),
}

# Arms that append the written file's state to the step's own output.
WRITE_THROUGH_ARMS = (InterventionArm.ARM_9_WRITE_THROUGH, InterventionArm.ARM_10_WRITE_THROUGH_DIFF,
                      InterventionArm.ARM_12_DEPENDENCY_WRITE_THROUGH,
                      InterventionArm.ARM_13_WRITE_THROUGH_DIFF_LEAN, InterventionArm.ARM_14_WRITE_THROUGH_DIFF_LINES)
# Write-through arms whose block is the net diff since the last view.
DIFF_ARMS = (InterventionArm.ARM_10_WRITE_THROUGH_DIFF, InterventionArm.ARM_12_DEPENDENCY_WRITE_THROUGH,
             InterventionArm.ARM_13_WRITE_THROUGH_DIFF_LEAN, InterventionArm.ARM_14_WRITE_THROUGH_DIFF_LINES)
# Arms that skip the block for writes that tell the agent nothing new (src/harness/lean.py).
LEAN_ARMS = (InterventionArm.ARM_13_WRITE_THROUGH_DIFF_LEAN, InterventionArm.ARM_14_WRITE_THROUGH_DIFF_LINES)


# Synced context (Arm 11). CORVUS injects whole files; a 32K-token backbone
# cannot hold that on SWE-Bench Pro's large files, so the block is capped and
# files are shown newest-viewed first, each clipped past SYNC_MAX_FILE_CHARS.
# Opens the block appended to the newest message. The prompt cache breakpoint
# goes just before it (PostQodeModel._payload), so the history is cached and
# only the block is recomputed: CORVUS's layout, history then synced context.
SYNC_HEADER = "[SYNCED CONTEXT: current contents of the files you have viewed]"
SYNC_MAX_CHARS = int(os.environ.get("SYNC_MAX_CHARS", 24_000))
SYNC_MAX_FILE_CHARS = int(os.environ.get("SYNC_MAX_FILE_CHARS", 12_000))
# Under a large window (over 64K tokens) the arm re-sends whole files, as
# CORVUS does. These limits only guard against a pathological file (a
# generated bundle); the largest block the v13 pilot runs would have sent
# uncapped was 51K characters. A step whose block hit them records it.
SYNC_WIDE_MAX_CHARS = 400_000
SYNC_WIDE_MAX_FILE_CHARS = 200_000
# An observation counts as "only a read" of a file when the file's text makes
# up at least this share of it; anything carrying more output is kept.
SYNC_READ_SHARE = 0.8
_OUTPUT_TAG = re.compile(r"<output>\n?(.*?)</output>", re.S)


def synced_context_view(messages: List[Dict[str, Any]], bindings: List[ObservationBinding],
                        workspace_root: str, max_chars: Optional[int] = None,
                        max_file_chars: Optional[int] = None) -> "tuple[List[Dict[str, Any]], List[str]]":
    """The prompt the synced-context arm sends, and the files its block showed.

    Returns a copy: the real history keeps every observation, as in CORVUS,
    where the synced context is rebuilt for each reasoning cycle and never
    stored in the trajectory.
    """
    transformed = copy.deepcopy(messages)
    registry: List[str] = []
    for b in sorted(bindings, key=lambda b: b.step_id, reverse=True):
        if b.file_path not in registry:
            registry.append(b.file_path)
    for b in bindings:
        # The newest read is elided too, as CORVUS's sync_file returns only a
        # marker: the file is in the block below, and the history stays
        # append-only, so the prompt cache can reuse it.
        if b.message_index >= len(transformed) or b.message_index < 0:
            continue
        content = str(transformed[b.message_index].get("content", ""))
        # mini-swe-agent wraps a command's output in <returncode>/<output> tags.
        inner = _OUTPUT_TAG.search(content)
        printed = inner.group(1) if inner else content
        body = b.file_content if b.file_content is not None else ""
        if body and len(body) >= SYNC_READ_SHARE * len(printed):
            transformed[b.message_index]["content"] = (
                f"[Contents of {b.file_path} are kept current in the synced context at the end of the prompt.]")
    max_chars = SYNC_MAX_CHARS if max_chars is None else max_chars
    max_file_chars = SYNC_MAX_FILE_CHARS if max_file_chars is None else max_file_chars
    blocks, shown, used = [], [], 0
    for rel in registry:
        path = rel if os.path.isabs(rel) else os.path.join(workspace_root, rel)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue  # deleted or unreadable: nothing current to show
        if len(text) > max_file_chars:
            text = text[:max_file_chars] + f"\n...[{len(text) - max_file_chars} more characters]"
        block = f"=== {rel} ===\n{text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        shown.append(rel)
        used += len(block)
    if blocks and transformed and transformed[-1].get("role") in ("user", "tool"):
        transformed[-1]["content"] = (str(transformed[-1].get("content", "")) +
                                      "\n\n" + SYNC_HEADER + "\n" +
                                      "\n\n".join(blocks))
    return transformed, shown


def _evict_oldest_by_recency(
    transformed: List[Dict[str, Any]],
    budget: int,
    min_tokens: int,
    placeholder_template: str,
    skip_indices: Optional[Dict[int, ObservationBinding]] = None,
) -> None:
    """Evict oldest eligible user/tool messages in place until budget tokens are freed."""
    skip_indices = skip_indices or {}
    evicted_tokens = 0
    for i, msg in enumerate(transformed):
        if i in skip_indices or msg.get("role") not in ["user", "tool"] or i <= 1:
            continue
        content = msg.get("content", "")
        tok_len = len(content) // 4
        if tok_len > min_tokens:
            msg["content"] = placeholder_template.format(tok_len=tok_len)
            evicted_tokens += tok_len
        if evicted_tokens >= budget:
            break


def apply_intervention_arm(
    messages: List[Dict[str, Any]],
    bindings: List[ObservationBinding],
    arm: InterventionArm,
    max_tokens: int = 32000,
    current_tokens: int = 0,
    workspace_root: str = "."
) -> List[Dict[str, Any]]:
    """Transform conversation message history according to selected intervention arm."""
    transformed = copy.deepcopy(messages)

    if arm in [InterventionArm.ARM_1_CONTROL, InterventionArm.ARM_0_PROMPT_ONLY,
               InterventionArm.ARM_9_WRITE_THROUGH, InterventionArm.ARM_10_WRITE_THROUGH_DIFF,
               InterventionArm.ARM_11_SYNCED_CONTEXT, InterventionArm.ARM_12_DEPENDENCY_WRITE_THROUGH,
               InterventionArm.ARM_13_WRITE_THROUGH_DIFF_LEAN, InterventionArm.ARM_14_WRITE_THROUGH_DIFF_LINES]:
        # Arm 11 builds its prompt with synced_context_view, which needs the
        # workspace; here, as for the others, the history is left unchanged.
        # Re-read reminder, Control, Write-through viewport:
        # Return unmodified history in context. Arm 9's intervention operates
        # at the tool-output layer (viewport echo after writes), not by
        # retroactively rewriting prompt history.
        return transformed

    elif arm == InterventionArm.ARM_2_POSITIONAL:
        # Recency eviction: oldest observations first, stale or not
        # Evict oldest observation contents when context exceeds limit
        if current_tokens > max_tokens:
            excess = current_tokens - max_tokens
            _evict_oldest_by_recency(
                transformed, excess, 100,
                "[CONTENT ELIDED BY RECENCY: {tok_len} tokens truncated]"
            )
        return transformed

    elif arm == InterventionArm.ARM_3_MARKED:
        # Stale flag
        # Inject inline warning if an observation has been invalidated
        stale_indices = {b.message_index: b for b in bindings if b.is_invalid}
        for idx, binding in stale_indices.items():
            if idx < len(transformed):
                orig_content = transformed[idx].get("content", "")
                marker = (
                    f"\n[SYSTEM NOTICE: File '{binding.file_path}' has been modified on disk "
                    f"at Step {binding.invalidation_step}. The observation above is STALE.]"
                )
                if marker not in orig_content:
                    transformed[idx]["content"] = orig_content + marker
        return transformed

    elif arm == InterventionArm.ARM_4_MASKED:
        # Stale-view removal
        # Replace stale observation entirely with structural placeholder
        stale_indices = {b.message_index: b for b in bindings if b.is_invalid}
        for idx, binding in stale_indices.items():
            if idx < len(transformed):
                placeholder = (
                    f"[OBSERVATION EVICTED: Content of '{binding.file_path}' observed at Step {binding.step_id} "
                    f"is STALE following on-disk modification at Step {binding.invalidation_step}. "
                    f"Execute 'cat {binding.file_path}' to inspect current on-disk state.]"
                )
                transformed[idx]["content"] = placeholder
        return transformed

    elif arm == InterventionArm.ARM_5_VOLUME_CONTROL:
        # Volume-matched eviction (oldest-first, from non-stale observations)
        # Calculate tokens masked in Arm 4, but evict equivalent tokens from NON-STALE observations
        stale_indices = {b.message_index: b for b in bindings if b.is_invalid}
        stale_tokens = sum(len(b.raw_content) // 4 for b in stale_indices.values())

        if stale_tokens > 0:
            _evict_oldest_by_recency(
                transformed, stale_tokens, 50,
                "[VOLUME CONTROL EVICTED: {tok_len} tokens removed without regard to staleness]",
                skip_indices=stale_indices,
            )
        return transformed

    elif arm == InterventionArm.ARM_6_IN_PLACE:
        # In-place refresh
        # Replaces stale observations in-place with current on-disk file state
        stale_indices = {b.message_index: b for b in bindings if b.is_invalid}
        for idx, binding in stale_indices.items():
            if idx < len(transformed):
                abs_p = os.path.join(workspace_root, binding.file_path) if not os.path.isabs(binding.file_path) else binding.file_path
                if os.path.exists(abs_p) and os.path.isfile(abs_p):
                    try:
                        with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                            fresh_content = f.read()
                        transformed[idx]["content"] = (
                            f"[OBSERVATION SYNCHRONIZED (Step {binding.invalidation_step}): "
                            f"Content of '{binding.file_path}' updated to current on-disk state]:\n"
                            f"{fresh_content}"
                        )
                    except Exception as e:
                        transformed[idx]["content"] = (
                            f"[OBSERVATION SYNCHRONIZED: File '{binding.file_path}' "
                            f"could not be re-read: {e}]"
                        )
                else:
                    transformed[idx]["content"] = (
                        f"[OBSERVATION SYNCHRONIZED: File '{binding.file_path}' "
                        f"was removed on disk at Step {binding.invalidation_step}.]"
                    )
        return transformed

    elif arm == InterventionArm.ARM_7_JIT_PREFETCH:
        # Tail refresh: replaces stale views AND appends current content, so it
        # rewrites earlier turns too; it is not append-only
        # 1. Mask the old stale observation to prevent reading superseded content
        stale_indices = {b.message_index: b for b in bindings if b.is_invalid}
        for idx, binding in stale_indices.items():
            if idx < len(transformed):
                transformed[idx]["content"] = (
                    f"[OBSERVATION SUPERSEDED: Content of '{binding.file_path}' observed at Step {binding.step_id} "
                    f"is STALE after Step {binding.invalidation_step}. (Synchronized snapshot forwarded below).]"
                )

        # 2. Forward the latest on-disk state of all mutated files at the immediate prompt tail
        if stale_indices and transformed:
            forwarding_blocks = []
            # Deduplicate by file path, keeping latest invalidation step
            latest_by_file: Dict[str, ObservationBinding] = {}
            for b in bindings:
                if b.is_invalid:
                    latest_by_file[b.file_path] = b

            for f_path, binding in latest_by_file.items():
                abs_p = os.path.join(workspace_root, f_path) if not os.path.isabs(f_path) else f_path
                if os.path.exists(abs_p) and os.path.isfile(abs_p):
                    try:
                        with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                            f_content = f.read()
                        forwarding_blocks.append(
                            f"[HARNESS STORE-FORWARDING BUFFER: File '{f_path}' (mutated at Step {binding.invalidation_step}) current on-disk content]:\n"
                            f"{f_content}"
                        )
                    except Exception:
                        pass

            if forwarding_blocks:
                forwarding_text = "\n\n" + "\n\n".join(forwarding_blocks)
                # Append to the latest message if it is a user observation, or append a new user message
                if transformed[-1].get("role") in ["user", "tool"]:
                    transformed[-1]["content"] += forwarding_text
                else:
                    transformed.append({"role": "user", "content": forwarding_text.strip()})

        return transformed

    elif arm == InterventionArm.ARM_8_FINE_GRAINED:
        # Changed-hunk removal
        # Instead of evicting the entire observation block, only invalidate line hunks that mutated on disk
        stale_indices = {b.message_index: b for b in bindings if b.is_invalid}
        for idx, binding in stale_indices.items():
            if idx < len(transformed):
                abs_p = os.path.join(workspace_root, binding.file_path) if not os.path.isabs(binding.file_path) else binding.file_path
                if os.path.exists(abs_p) and os.path.isfile(abs_p):
                    try:
                        with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                            fresh_content = f.read()

                        # Diff the file as it was read against the file as it
                        # is now. Using the observation message here compared
                        # harness text with source and marked every line
                        # changed, which made this arm identical to Arm 4.
                        # None (never read) still falls back to raw_content;
                        # "" (read, genuinely empty) must not.
                        orig_content = binding.file_content if binding.file_content is not None else binding.raw_content
                        orig_lines = orig_content.splitlines(keepends=True)
                        fresh_lines = fresh_content.splitlines(keepends=True)
                        matcher = difflib.SequenceMatcher(None, orig_lines, fresh_lines)

                        reconstructed_blocks = []
                        for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
                            if tag == "equal":
                                reconstructed_blocks.append("".join(orig_lines[alo:ahi]))
                            elif tag in ("replace", "delete"):
                                hunk_desc = f"{alo + 1}" if alo + 1 == ahi else f"{alo + 1}-{ahi}"
                                reconstructed_blocks.append(
                                    f"[HUNK INVALIDATED (Observed lines {hunk_desc}): modified on disk at Step {binding.invalidation_step}]\n"
                                )
                            elif tag == "insert":
                                num_lines = bhi - blo
                                reconstructed_blocks.append(
                                    f"[HUNK INSERTED on disk at Step {binding.invalidation_step} (+{num_lines} lines)]\n"
                                )

                        transformed[idx]["content"] = "".join(reconstructed_blocks)
                    except Exception as e:
                        transformed[idx]["content"] = (
                            f"[OBSERVATION EVICTED: Content of '{binding.file_path}' observed at Step {binding.step_id} "
                            f"is STALE (read error: {e})]"
                        )
                else:
                    transformed[idx]["content"] = (
                        f"[OBSERVATION EVICTED: Content of '{binding.file_path}' observed at Step {binding.step_id} "
                        f"is STALE (file removed on disk)]"
                    )
        return transformed

    return transformed
