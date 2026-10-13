"""Domain-specific corruption oracles for coding agents (RQ2 / RQ3).

Provides code-specific evaluation metrics:
1. AST Parseability: Verifies syntax validity of touched code.
2. Blast Radius: Measures scope-drift footprint against ground-truth touched files.
3. Corruption Spikes: Flags single-step catastrophic deviance (per DELEGATE-52).
"""

import ast
import re
from typing import List, Set, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class ParseabilityResult:
    is_valid: bool
    language: str
    error_message: str = ""
    error_line: int = 0


@dataclass
class BlastRadiusResult:
    ground_truth_count: int
    touched_count: int
    unrelated_touched_count: int
    blast_radius_ratio: float
    unrelated_files: List[str] = field(default_factory=list)


def check_ast_parseability(code_content: str, filepath: str = "") -> ParseabilityResult:
    """Verify syntax validity of source code across Python and common languages."""
    if not code_content.strip():
        return ParseabilityResult(is_valid=True, language="unknown")

    # Python syntax parsing
    if filepath.endswith(".py"):
        try:
            ast.parse(code_content)
            return ParseabilityResult(is_valid=True, language="python")
        except SyntaxError as e:
            return ParseabilityResult(
                is_valid=False,
                language="python",
                error_message=str(e.msg),
                error_line=e.lineno or 0
            )

    # Basic brace/bracket balance check for C/C++/Rust/JS/TS
    # Strip strings and line comments for a simple syntactic integrity heuristic
    # 1. Strip multi-line comments /* ... */
    no_block_comments = re.sub(r'/\*.*?\*/', '', code_content, flags=re.DOTALL)

    # 2. Strip single-line comments //...
    lines = no_block_comments.split("\n")
    cleaned_lines = []
    for line in lines:
        line_no_comment = re.sub(r'//.*$', '', line)
        cleaned_lines.append(line_no_comment)
    
    full_cleaned = "\n".join(cleaned_lines)
    # 3. Normalize single-character literals (e.g. '(', ')') so they do not pollute the stack
    full_cleaned = re.sub(r"'\\?.'", "''", full_cleaned)
    stack = []
    brackets = {')': '(', '}': '{', ']': '['}
    in_str = False
    str_char = ''

    for idx, ch in enumerate(full_cleaned):
        if ch in ['"', "'", '`'] and (idx == 0 or full_cleaned[idx-1] != '\\'):
            if not in_str:
                in_str = True
                str_char = ch
            elif str_char == ch:
                in_str = False
        elif not in_str:
            if ch in brackets.values():
                stack.append(ch)
            elif ch in brackets.keys():
                if not stack or stack.pop() != brackets[ch]:
                    return ParseabilityResult(
                        is_valid=False,
                        language="c_family_or_rust",
                        error_message=f"Unmatched bracket: {ch}",
                        error_line=0
                    )

    if stack:
        return ParseabilityResult(
            is_valid=False,
            language="c_family_or_rust",
            error_message=f"Unclosed opening bracket(s): {stack}",
            error_line=0
        )

    return ParseabilityResult(is_valid=True, language="general")


def calculate_blast_radius(touched_files: Set[str], ground_truth_files: Set[str]) -> BlastRadiusResult:
    """Compute the blast radius: scope drift beyond ground truth touched files."""
    gt_clean = {f.removeprefix("./") for f in ground_truth_files if f}
    touched_clean = {f.removeprefix("./") for f in touched_files if f}

    if not gt_clean:
        return BlastRadiusResult(
            ground_truth_count=0,
            touched_count=len(touched_clean),
            unrelated_touched_count=len(touched_clean),
            blast_radius_ratio=float(len(touched_clean)),
            unrelated_files=list(touched_clean)
        )

    unrelated = touched_clean - gt_clean
    ratio = len(touched_clean) / max(1, len(gt_clean))

    return BlastRadiusResult(
        ground_truth_count=len(gt_clean),
        touched_count=len(touched_clean),
        unrelated_touched_count=len(unrelated),
        blast_radius_ratio=round(ratio, 3),
        unrelated_files=sorted(list(unrelated))
    )


def detect_corruption_spikes(step_deviations: List[float], spike_threshold: float = 2.0) -> List[int]:
    """Detect single-step corruption spikes per DELEGATE-52 (sudden jumps vs running baseline)."""
    spikes = []
    if len(step_deviations) < 3:
        return spikes

    for i in range(1, len(step_deviations)):
        baseline = sum(step_deviations[:i]) / max(1, i)
        val = step_deviations[i]
        # Spike detected if single step deviance exceeds baseline by factor of threshold
        if val > 0 and (val - baseline) >= (spike_threshold * max(0.1, baseline)):
            spikes.append(i + 1)  # 1-indexed step

    return spikes
