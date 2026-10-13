"""Coherence metrics: state writes, and the mismatches they cause.

One replay implementation, used for both research questions, so that a hazard
means the same thing everywhere. Previously it did not: the RQ1 detector
cleared a hazard when the agent re-read the file, while the RQ2 hook marked a
binding invalid forever, and the headline RQ2 number summed live-invalid
bindings at every step. That sum grows with trajectory length and with edit
count, so ranking the intervention arms by it put the arms that edited least
on top, and charged the arms that repair the context for the hazards they
repaired.

The metrics are grouped so each answers one question, and each has a stated
denominator:

  L0  activity        how much the agent actually did
  L1  incidence       how often it acted on a view it knew was stale
  L2  exposure        how much stale text the prompt still carried
  L3  mismatch        whether the divergence produced a wrong action
  L4  outcome         whether the task was solved (computed elsewhere)

L1 is a property of behaviour and is comparable across arms. L2 is a property
of the intervention and is mostly a manipulation check: an arm that claims to
withdraw stale text should score zero, and if it does not, the arm is not
being applied. L3 is the one that says whether staleness cost anything.
"""

import re
import shlex
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from src.harness.arms import ARM_REMEDIATION, ArmRemediation, InterventionArm
from src.metrics.replay import live_steps, replay

# Commands whose job is to change a file. Matching is on the command verb,
# not on the text of the line: a substring test counts `cat patch.txt` as a
# mutation because of the word "patch", which inflated the edit-attempt count
# on the wave-1 traces from roughly 400 to 627.
EDIT_VERBS = {
    "tee", "patch", "cp", "mv", "rm", "truncate", "install", "dd",
    "shred", "ln", "rsync",
}
# Verbs that only edit with the right flag.
INPLACE_FLAG_VERBS = {"sed": ("-i",), "perl": ("-i",), "ruby": ("-i",)}
# git subcommands that write patches or modify files as an intended code edit.
# Navigation and revert commands (checkout, restore, reset, clean) are not code authoring edits.
GIT_WRITE_SUBCOMMANDS = {"apply", "mv", "rm"}
# A python one-liner or heredoc that opens a file for writing (excluding read-only 'r' and 'rb').
PY_WRITE_RE = re.compile(r"""open\s*\([^)]*['"](?:[wax]\+?b?|r\+b?)['"]""")
SHELL_SEPARATORS = re.compile(r"(?:\|\||&&|[;|&\n])")


def _has_output_redirection(cmd: str) -> bool:
    """Whether command has an unquoted output redirection (e.g. > or >>).

    Respects quotes so comparison operators inside Python one-liners or
    grep patterns (e.g. 'if x > 1:') are not misclassified as redirections.
    """
    try:
        lexer = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lexer.whitespace_split = False
        lexer.commenters = '#'
        tokens = list(lexer)
    except Exception:
        tokens = cmd.split()

    for i, tok in enumerate(tokens):
        if tok in ('>', '>>') or (tok.startswith('>') and not tok.startswith('>&')):
            # Ensure it is not stderr redirection like 2>&1
            if i > 0 and tokens[i-1] in ('2', '2>'):
                continue
            return True
        if tok.endswith('>') and not tok.endswith('2>') and not tok.endswith('&>'):
            return True
    return False


def _simple_commands(command: str) -> List[str]:
    """Split a command line into the simple commands it runs."""
    return [part.strip() for part in SHELL_SEPARATORS.split(command or "") if part.strip()]


def _verb_of(simple_command: str) -> Optional[List[str]]:
    """Tokens of a simple command with leading VAR=value assignments removed."""
    try:
        tokens = shlex.split(simple_command, comments=True)
    except ValueError:
        tokens = simple_command.split()
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens = tokens[1:]
    return tokens or None


def is_edit_attempt(command: str) -> bool:
    """Whether a command is trying to change a file on disk.

    Intent, not effect: that is the point. A command that intends to edit and
    changes nothing is the signal we are looking for.
    """
    text = command or ""
    if PY_WRITE_RE.search(text):
        return True
    if _has_output_redirection(text):
        return True
    for simple in _simple_commands(text):
        tokens = _verb_of(simple)
        if not tokens:
            continue
        verb = tokens[0].rsplit("/", 1)[-1]
        if verb in EDIT_VERBS:
            return True
        if verb in INPLACE_FLAG_VERBS:
            flags = INPLACE_FLAG_VERBS[verb]
            if any(t.startswith(flags) for t in tokens[1:]):
                return True
        if verb.startswith("git") and len(tokens) > 1 and tokens[1] in GIT_WRITE_SUBCOMMANDS:
            return True
    return False


@dataclass
class _Binding:
    """The agent's current view of one file."""

    step: int
    content: str
    valid: bool = True
    invalidated_at: Optional[int] = None


@dataclass
class CoherenceMetrics:
    """Per-trajectory metrics. Rates carry their denominator in the name."""

    # L0 activity
    steps: int = 0
    reads: int = 0
    writes: int = 0
    edit_attempts: int = 0
    files_read: int = 0
    files_written: int = 0

    # L1 incidence
    invalidations: int = 0
    raw_hazards: int = 0
    hazard_opportunities: int = 0          # writes to a file the agent had read
    stale_act_rate: float = 0.0            # raw_hazards / hazard_opportunities
    hazards_per_step: float = 0.0
    # The same incidence counted a second way, crediting a harness-supplied
    # refresh as a re-read. The strict count above asks whether the agent
    # re-read the file itself; this one asks whether the agent could see
    # current content when it wrote, whoever fetched it. They differ only for
    # arms that put current content in the prompt (forwarding at the tail, or
    # refreshing in place), and the gap is the part of an arm's score that is
    # bookkeeping rather than behaviour.
    raw_hazards_forward_credited: int = 0
    stale_act_rate_forward_credited: float = 0.0
    # Secondary: whole-file rewrites of a seen file, and those that landed on
    # a version the agent had not seen (see replay.ReplayResult).
    overwrite_opportunities: int = 0
    lost_updates: int = 0
    at_risk_opportunities: int = 0
    at_risk_hazards: int = 0

    # L2 exposure
    stale_context_fraction: float = 0.0    # mean over steps of visible stale bytes / bound observation bytes
    coherent_prompt_rate: float = 1.0      # steps sent with no visible stale bytes
    peak_stale_bytes: int = 0

    # L3 mismatch
    phantom_edits: int = 0                 # edit intent, nothing changed on disk
    failed_commands: int = 0
    stale_anchored_edits: int = 0
    sae_per_edit_attempt: float = 0.0
    failure_rate_when_stale: Optional[float] = None
    failure_rate_when_fresh: Optional[float] = None
    failure_lift_under_staleness: Optional[float] = None
    commitment_tax_steps: int = 0
    commitment_tax_fraction: float = 0.0
    first_stale_anchored_step: Optional[int] = None

    # bookkeeping
    exact_sae: bool = False                # True when live per-edit capture was available
    hazard_files: List[str] = field(default_factory=list)
    # How many of the agent's views the harness recorded, which bounds how
    # much of the agent's context any arm could act on.
    agent_views: int = 0
    harness_bound_views: int = 0


def _visible_stale_bytes(bindings: Dict[str, _Binding], remediation: ArmRemediation) -> int:
    """Bytes of stale observation text the arm still shows the model.

    An arm that withdraws stale content shows none of it, which is what makes
    this a check on the intervention rather than a measure of the agent.
    """
    if remediation.withdraws and not remediation.partial:
        return 0
    total = sum(len(b.content) for b in bindings.values() if not b.valid)
    if remediation.partial:
        # Only the changed region is withdrawn. Without the diff we cannot say
        # how much that was, so charge half and say so in the paper rather
        # than pretend the arm withdrew everything.
        return total // 2
    return total


def compute(steps: Sequence[Dict[str, Any]], arm: str) -> CoherenceMetrics:
    """Replay one trajectory's steps into coherence metrics.

    `steps` are trace step records, which carry the filesystem-validated
    `reads_detected` and `writes_detected` for each step, the command, its
    return code, and the observation text.
    """
    try:
        remediation = ARM_REMEDIATION[InterventionArm(arm)]
    except (KeyError, ValueError):
        remediation = ArmRemediation()

    m = CoherenceMetrics(steps=len(steps))
    bindings: Dict[str, _Binding] = {}
    read_files: Set[str] = set()
    written_files: Set[str] = set()
    fractions: List[float] = []
    fail_stale = fail_fresh = n_stale = n_fresh = 0
    exact_sae_seen = False

    for step in steps:
        reads = set(step.get("reads_detected") or [])
        writes = set(step.get("writes_detected") or [])
        command = step.get("command") or step.get("action") or ""
        observation = str(step.get("observation") or "")
        returncode = step.get("returncode")
        failed = returncode not in (0, None)
        edit_intent = is_edit_attempt(command)

        # --- state as the step begins: what the harness believed before acting
        stale_now = {f for f, b in bindings.items() if not b.valid}
        visible = _visible_stale_bytes(bindings, remediation)
        # Denominator is the bytes of file observations the agent is holding,
        # which is the cache whose coherence is in question. Measuring against
        # the whole prompt instead buries the signal under the task statement
        # and the accumulated command echoes.
        bound_bytes = sum(len(b.content) for b in bindings.values())
        fractions.append(visible / bound_bytes if bound_bytes else 0.0)
        m.peak_stale_bytes = max(m.peak_stale_bytes, visible)

        # --- mismatch signals
        if edit_intent:
            m.edit_attempts += 1
            if not writes:
                m.phantom_edits += 1
        if failed:
            m.failed_commands += 1

        # Exact detection when the run recorded it; otherwise infer.
        bound_matches = step.get("bound_matches_disk")
        if isinstance(bound_matches, dict) and bound_matches:
            exact_sae_seen = True
            # Only the files this step touched: the harness records a match
            # for every file ever bound, and a stale file the edit never names
            # says nothing about whether this edit was anchored on a stale view.
            touched = reads | writes
            if edit_intent and any(v is False for f, v in bound_matches.items() if f in touched):
                m.stale_anchored_edits += 1
                if m.first_stale_anchored_step is None:
                    m.first_stale_anchored_step = step.get("step_id")
        elif edit_intent and (failed or not writes) and (stale_now & (reads | writes)):
            m.stale_anchored_edits += 1
            if m.first_stale_anchored_step is None:
                m.first_stale_anchored_step = step.get("step_id")

        if stale_now:
            n_stale += 1
            fail_stale += failed
        else:
            n_fresh += 1
            fail_fresh += failed

        # --- apply the step: reads rebind, writes invalidate
        read_share = observation[: len(observation) // len(reads)] if reads else observation
        for f in reads:
            bindings[f] = _Binding(step=step.get("step_id", 0), content=read_share)
            read_files.add(f)
            m.reads += 1
        for f in writes:
            written_files.add(f)
            m.writes += 1
            b = bindings.get(f)
            if b is not None and b.valid:
                b.valid = False
                b.invalidated_at = step.get("step_id")
                m.invalidations += 1

    m.files_read = len(read_files)
    m.files_written = len(written_files)
    m.exact_sae = exact_sae_seen
    m.stale_context_fraction = statistics.mean(fractions) if fractions else 0.0

    # Incidence and delivered coherence come from the replay, which orders
    # each step's reads and writes, scores only anchored edits, and credits an
    # arm's supplied content per file. See src/metrics/replay.py.
    rr = replay(live_steps(steps), withdraws=remediation.withdraws,
                forwards=remediation.forwards, refreshes=remediation.refreshes)
    m.raw_hazards = rr.hazards_strict
    m.raw_hazards_forward_credited = rr.hazards_credited
    m.hazard_opportunities = rr.opportunities
    m.overwrite_opportunities = rr.overwrite_opportunities
    m.lost_updates = rr.lost_updates
    m.at_risk_opportunities = rr.at_risk_opportunities
    m.at_risk_hazards = rr.at_risk_hazards
    m.hazard_files = rr.hazard_files
    m.coherent_prompt_rate = rr.coherent_steps / len(steps) if steps else 1.0
    m.agent_views = rr.agent_views
    m.harness_bound_views = rr.harness_bound_views
    m.stale_act_rate = m.raw_hazards / m.hazard_opportunities if m.hazard_opportunities else 0.0
    m.stale_act_rate_forward_credited = (
        m.raw_hazards_forward_credited / m.hazard_opportunities
        if m.hazard_opportunities else 0.0
    )
    m.hazards_per_step = m.raw_hazards / len(steps) if steps else 0.0
    m.sae_per_edit_attempt = m.stale_anchored_edits / m.edit_attempts if m.edit_attempts else 0.0
    m.failure_rate_when_stale = (fail_stale / n_stale) if n_stale else None
    m.failure_rate_when_fresh = (fail_fresh / n_fresh) if n_fresh else None
    if m.failure_rate_when_stale is not None and m.failure_rate_when_fresh is not None:
        m.failure_lift_under_staleness = m.failure_rate_when_stale - m.failure_rate_when_fresh
    if m.first_stale_anchored_step is not None and steps:
        last = steps[-1].get("step_id", len(steps))
        m.commitment_tax_steps = max(0, last - m.first_stale_anchored_step + 1)
        m.commitment_tax_fraction = m.commitment_tax_steps / len(steps)
    return m


def scope_scores(modified: Iterable[str], gold: Iterable[str]) -> Dict[str, float]:
    """Precision, recall and F1 of the files changed against the gold patch.

    The ratio this replaces, |modified| / |gold|, scores a perfect 1.0 for an
    agent that edited one file when the gold patch edits one file, even when
    it is not the same file. Precision and recall cannot be satisfied that way,
    and recall is what exposes an arm that "improves" by doing less.
    """
    mod, gt = set(modified), set(gold)
    hit = len(mod & gt)
    precision = hit / len(mod) if mod else 0.0
    recall = hit / len(gt) if gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "scope_precision": round(precision, 4),
        "scope_recall": round(recall, 4),
        "scope_f1": round(f1, 4),
        "extra_files": len(mod - gt),
        "missed_gold_files": len(gt - mod),
    }
