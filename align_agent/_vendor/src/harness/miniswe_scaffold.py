"""Lightweight MiniSWE-Agent scaffold with pluggable freshness intervention arms."""

import os
import re
import subprocess
import logging
from typing import List, Dict, Any, Optional, Tuple
from .freshness_hook import FreshnessHook
from .arms import DIFF_ARMS, PROMPT_ONLY_POLICY, WRITE_THROUGH_ARMS, InterventionArm, apply_intervention_arm
from .viewport import diff_snapshots as _viewport_diff_snapshots
from .viewport import snapshot_workspace as _viewport_snapshot_workspace
from .viewport import viewport_blocks
from .tail_diff import tail_diff_blocks
from src.parsers.base_parser import BaseTrajectoryParser, ActionKind
from src.parsers.file_events import CONTAINER_ROOTS
from src.parsers.miniswe_parser import MiniSWEParser

logger = logging.getLogger(__name__)


class MiniSWEScaffold:
    """Lightweight bash-only coding agent scaffold equipped with freshness intervention arms."""

    def __init__(
        self,
        arm: InterventionArm = InterventionArm.ARM_1_CONTROL,
        model_name: str = "qwen3.5-9b-mlx",
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        workspace_dir: str = ".",
        max_steps: int = 30,
        max_tokens: int = 32000,
        use_mtime_cache: bool = True,
        parser_cls: type = MiniSWEParser,
    ):
        self.arm = arm
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.use_mtime_cache = use_mtime_cache

        # Instrumentation (observe_step, extract_bash_file_operations) is
        # inherited from BaseTrajectoryParser and identical regardless of
        # which framework's parser subclass is used for shell-text commands;
        # the parameter exists so a non-mini-SWE-agent scaffold reusing this
        # class purely as an instrumentation toolbox can state its real
        # provenance instead of silently claiming MiniSWEParser's.
        self.parser = parser_cls()
        self.freshness_hook = FreshnessHook(workspace_root=self.workspace_dir, use_mtime_cache=self.use_mtime_cache)
        self.messages: List[Dict[str, Any]] = []
        self.step_counter = 0

    TERMINATION_TOKEN = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    # Fence info strings that mark a block as a shell command.
    ACTION_TAGS = {"bash", "sh", "shell"}
    # The token alone on a line, optionally echoed (the mini-swe-agent idiom).
    TOKEN_LINE_RE = re.compile(r"""^(?:echo\s+)?["']?COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT["']?$""")

    def _fenced_blocks(self, text: str) -> List[Tuple[str, str, int, bool]]:
        """Fenced blocks as (info, body, end_offset, closed), fences paired line by line.

        A regex scan paired fences wrongly as soon as a response contained a
        non-shell block: the closing fence of a quoted ```python snippet was
        taken as the opening of a new block, so the prose between the snippet
        and the real bash block was extracted and run as a command, and the
        bash block's own opening fence was consumed as a closer. It also read
        ```shell as ```sh followed by a command beginning "ell". An unclosed
        block (a response cut off by the token limit) runs to the end of text.
        """
        blocks: List[Tuple[str, str, int, bool]] = []
        pos = 0
        info: Optional[str] = None
        body: List[str] = []
        for line in text.splitlines(keepends=True):
            s = line.strip()
            end = pos + len(line)
            if info is None:
                if s.startswith("```"):
                    rest = s[3:]
                    if "```" in rest:
                        # One-line form: ```bash ls -la```
                        inner = rest[: rest.index("```")].strip()
                        tag, _, cmd = inner.partition(" ")
                        if tag.lower() in self.ACTION_TAGS:
                            blocks.append((tag.lower(), cmd, end, True))
                        else:
                            blocks.append(("", inner, end, True))
                    else:
                        words = rest.strip().lower().split()
                        info = words[0] if words else ""
                        body = []
            elif s == "```":
                blocks.append((info, "".join(body), end, True))
                info = None
            else:
                body.append(line)
            pos = end
        if info is not None:
            blocks.append((info, "".join(body), pos, False))
        return blocks

    def _action_block(self, text: str) -> Optional[Tuple[str, int]]:
        """The turn's one action: its first non-empty shell block, else its first bare block.

        Returns (command, end_offset) or None when the turn has no fenced block.
        """
        blocks = self._fenced_blocks(text)
        for wanted in (lambda i: i in self.ACTION_TAGS, lambda i: i == ""):
            for info, body, end, _closed in blocks:
                if wanted(info) and body.strip():
                    return body.strip(), end
        return None

    def has_complete_action(self, text: str) -> bool:
        """Whether a partial response already holds a closed shell-tagged block.

        Used to stop generation early. Only a tagged block qualifies: a bare
        block may be a code quote with the real command still to come. The
        closing fence must be followed by a newline, since a trailing ``` in a
        stream may still grow into an opening fence such as ```python.
        """
        if "```" not in text:
            return False
        for info, body, end, closed in self._fenced_blocks(text):
            if info in self.ACTION_TAGS and body.strip():
                return closed and text[end - 1] == "\n"
        return False

    def truncate_to_action(self, raw_text: str) -> str:
        """Cut a response off after its action block.

        Some models do not stop after issuing a command. They go on to write
        the output they expect and further commands that depend on it: one
        model produced 20 to 36 bash blocks per turn until the 4096-token cap.
        Running all of them executes edits decided on without seeing any real
        output, which is not agent behaviour, and the imagined outputs would
        sit in the context as if observed. As in mini-swe-agent, a turn is one
        action; whatever follows it is discarded and never enters the context.
        """
        text = raw_text or ""
        hit = self._action_block(text)
        if hit is None:
            return text
        return text[: hit[1]].rstrip()

    def extract_action_command(self, raw_text: str) -> str:
        """Extract executable bash command from LLM response (handling markdown code fences and thoughts)."""
        text = raw_text.strip()
        if not text:
            return ""

        # Code fences are the agent's actual instructions, so they are read
        # before anything else. The termination token is only honoured below,
        # once it is known whether real commands were issued: checking it
        # against the whole response first meant a model that merely mentioned
        # the token while planning had its work discarded and its trajectory
        # ended. One observed turn carried the working fix and a mention of the
        # token together, and the fix was thrown away.
        hit = self._action_block(text)
        if hit is not None:
            command_lines = [
                ln for ln in hit[0].splitlines()
                if not self.TOKEN_LINE_RE.match(ln.strip())
            ]
            remaining = "\n".join(command_lines).strip()
            if remaining:
                return remaining
            return self.TERMINATION_TOKEN

        if self.TERMINATION_TOKEN in text:
            return self.TERMINATION_TOKEN

        return self._fallback_command(text)

    def signals_completion(self, raw_text: str) -> bool:
        """Whether the turn genuinely requests termination.

        Only the action block containing the token on its own line counts, or
        the token appearing when the turn issued no commands at all. A passing
        mention in prose must not end the trajectory and discard the work.
        """
        text = (raw_text or "").strip()
        if not text:
            return False
        hit = self._action_block(text)
        if hit is not None:
            return any(self.TOKEN_LINE_RE.match(ln.strip()) for ln in hit[0].splitlines())
        return self.TERMINATION_TOKEN in text

    def _fallback_command(self, text: str) -> str:
        """Last-resort extraction for turns that used no code fence."""

        # Check for explicit shell prompt commands ($ ...)
        dollar_cmds = [line[2:].strip() for line in text.split("\n") if line.strip().startswith("$ ")]
        if dollar_cmds:
            return "\n".join(dollar_cmds)

        # Check for standard shell utility commands
        common_shell_prefixes = (
            "cat ", "echo ", "ls", "cd ", "git ", "grep ", "find ", "sed ",
            "python", "pip ", "pytest", "touch ", "mkdir ", "cp ", "mv ", "rm ",
            "head ", "tail ", "export ", "which ", "diff "
        )
        cmd_lines = [
            line.strip() for line in text.split("\n")
            if line.strip().startswith(common_shell_prefixes)
            and not line.strip().startswith("```")
        ]
        if cmd_lines:
            return "\n".join(cmd_lines)

        # This message is appended to the conversation and is re-parsed on
        # later turns, so it must not contain a fenced code block: an example
        # fence here would be extracted as the literal command "..." and
        # executed, trapping the agent in a loop against its own error text.
        return "[Harness: No valid bash command detected. Issue commands inside a bash code fence.]"

    def init_task(self, system_prompt: str, problem_statement: str):
        """Initialize the agent conversation history."""
        effective_system_prompt = system_prompt
        if self.arm == InterventionArm.ARM_0_PROMPT_ONLY:
            effective_system_prompt = system_prompt + PROMPT_ONLY_POLICY
        self.messages = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": f"Task instructions:\n{problem_statement}\nSolve this issue using bash commands."}
        ]
        self.step_counter = 0
        self.freshness_hook = FreshnessHook(workspace_root=self.workspace_dir, use_mtime_cache=self.use_mtime_cache)

    SNAPSHOT_SKIP_DIRS = {
        ".git", "__pycache__", ".pytest_cache", ".tox", "node_modules", ".mypy_cache",
        # Build trees. An agent that runs `setup.py build_ext --inplace` to be
        # able to import the package under test rewrites hundreds of files
        # here; counting those as edits it chose to make put 40 generated
        # .c and .so files into one cell's touched set.
        "build", "dist", ".eggs", ".cache", "htmlcov", ".venv", "venv",
        # The task's provisioned environment, which is infrastructure rather
        # than anything the agent authored.
        ".venv-task",
    }
    # Compiled and generated output, which an agent produces by building, not
    # by editing.
    IGNORED_SUFFIXES = (
        ".so", ".pyd", ".dylib", ".o", ".a", ".obj", ".lib", ".dll",
        ".pyc", ".pyo", ".class", ".egg-link",
    )
    # Syntax that mutates a file, used to label a line as a write rather than
    # a read once its referenced paths are known.
    MUTATION_RE = re.compile(
        r"""(?:sed\s+-[a-zA-Z]*i|--in-place|>>?(?!&)|\btee\b|\bpatch\b|git\s+(?:checkout|apply|restore)"""
        r"""|\b(?:cp|mv|rm|truncate|install)\b|open\([^)]*['"][wa]\+?['"])"""
    )

    def snapshot_workspace(self) -> Dict[str, tuple]:
        """Map every workspace file to (mtime_ns, size). See src/harness/viewport.py."""
        return _viewport_snapshot_workspace(self.workspace_dir, skip_dirs=self.SNAPSHOT_SKIP_DIRS)

    def diff_snapshots(self, before: Dict[str, tuple], after: Dict[str, tuple]) -> List[str]:
        """Files created or modified between two snapshots."""
        return _viewport_diff_snapshots(before, after)

    def ordered_file_events(self, cmd: str) -> List[tuple]:
        """Sequence of (op, path) events in the order the command performs them.

        Steps here average roughly 50 command lines, so one step routinely
        inspects and edits the same file more than once. Collapsing a step into
        "all reads, then all writes" mislabels the common `R R W R` shape: the
        trailing read happens after the edit and is fresh, yet was being marked
        stale. A pure write-then-verify step was inverted outright. Ordering is
        recovered per line; which writes actually landed remains decided by the
        filesystem snapshot.
        """
        events: List[tuple] = []
        for line in (cmd or "").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Paths are resolved against the filesystem rather than matched
            # per tool, so grep/sed/awk/python all work without a regex each.
            # Tool-specific parsing failed here in practice: for
            # `sed -i "" "s/a/b/" pkg/mod.py` it captured `s/a/b/` as the
            # target, and grep was not treated as a read at all, leaving the
            # ordering empty.
            files = self.resolve_referenced_files(stripped)
            if not files:
                continue
            op = "W" if self.MUTATION_RE.search(stripped) else "R"
            for path in files:
                events.append((op, path))
        return events

    _CONTAINER_ROOTS = CONTAINER_ROOTS

    def _drop_unchanged(self, paths: List[str]) -> List[str]:
        """Drop paths whose bytes still match the harness's last hash of them.

        The snapshot diff compares (mtime, size), so a tool that rewrites a
        file without changing it (a Python str.replace that matched nothing,
        sed -i via temp-file-and-rename) looks like a write and would
        invalidate a still-accurate binding. Only files the harness has
        already hashed can be checked, which is exactly the set whose
        bindings a spurious write could invalidate; hashing is limited to
        those, so a build step that touches thousands of files costs nothing.
        """
        hook = self.freshness_hook
        kept = []
        for rel in paths:
            abs_path = os.path.join(hook.workspace_root, rel)
            prior = hook.stat_cache.get(abs_path)
            if prior is not None:
                try:
                    if hook._hash_bytes(abs_path) == prior[2]:
                        continue
                except OSError:
                    pass
            kept.append(rel)
        return kept

    def _normalize_paths(self, paths: List[str]) -> List[str]:
        """Reduce paths to deduplicated workspace-relative form.

        The same file named absolutely and relatively must map to one binding,
        and anything outside the workspace (scratch files under /tmp, system
        site-packages) is not part of the task's edit scope.
        """
        out: List[str] = []
        ws_real = os.path.realpath(self.workspace_dir)
        for p in paths:
            if not p:
                continue
            for root in self._CONTAINER_ROOTS:
                if p.startswith(root):
                    p = os.path.join(self.workspace_dir, p[len(root):])
                    break
            candidate = p if os.path.isabs(p) else os.path.join(self.workspace_dir, p)
            try:
                # Existence is the discriminator. Command text yields plenty of
                # tokens that merely look like paths ("we", "if", "Django",
                # "#"), and admitting them inflates the drift ratio with words
                # from the agent's prose.
                if not os.path.isfile(candidate):
                    continue
                rel = os.path.relpath(os.path.realpath(candidate), ws_real)
            except (OSError, ValueError):
                continue
            parts = rel.split(os.sep)
            if rel.startswith("..") or rel == ".":
                continue
            if any(p in self.SNAPSHOT_SKIP_DIRS or p.endswith(".egg-info") for p in parts):
                continue
            if rel.endswith(self.IGNORED_SUFFIXES):
                continue
            if rel not in out:
                out.append(rel)
        return out

    def _command_env(self) -> Dict[str, str]:
        """Environment for agent commands, with the workspace shadowing site-packages.

        Without this, `import django` inside a django task resolved to the
        interpreter's installed Django rather than the checkout under test. An
        agent duly inspected the installed copy, found the bug already fixed in
        that version, declared the task complete and terminated, never having
        touched the task repository. Prepending the workspace to PYTHONPATH
        makes the checkout win, and keeps agents from reading or editing real
        site-packages.
        """
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{self.workspace_dir}{os.pathsep}{existing}" if existing else self.workspace_dir
        # Keep .pyc files out of the workspace so they cannot be mistaken for
        # agent-authored changes by the write-detection snapshot.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def resolve_referenced_files(self, cmd: str) -> List[str]:
        """Workspace files named by a command, validated against the filesystem.

        Every whitespace/quote-delimited token is tested for existence rather
        than pattern-matched, so prose and shell fragments ("we", "{print",
        "make_bytes/,/^") cannot be mistaken for paths, and unusual read
        commands (grep -n, sed -n, awk, python -c) are covered without needing
        a regex per tool.
        """
        found: List[str] = []
        for raw in re.split(r"[\s;|&]+", cmd or ""):
            tok = raw.strip("'\"`,()<>").lstrip("@")
            if not tok or tok.startswith("-"):
                continue
            clean_tok = tok
            for root in self._CONTAINER_ROOTS:
                if clean_tok.startswith(root):
                    clean_tok = clean_tok[len(root):]
                    break
            candidate = clean_tok if os.path.isabs(clean_tok) else os.path.join(self.workspace_dir, clean_tok)
            try:
                if not os.path.isfile(candidate):
                    continue
                rel = os.path.relpath(os.path.realpath(candidate), os.path.realpath(self.workspace_dir))
            except (OSError, ValueError):
                continue
            if rel.startswith("..") or rel.split(os.sep)[0] in self.SNAPSHOT_SKIP_DIRS:
                continue
            if rel not in found:
                found.append(rel)
        return found

    def execute_bash_command(self, cmd: str, already_extracted: bool = False) -> str:
        """Execute a bash command in the workspace directory.

        `already_extracted` must be set when the caller has already run
        extract_action_command. Extracting twice is destructive: the second
        pass sees no code fence, falls through to the line-prefix filter, and
        keeps only lines beginning with a shell keyword. That silently shreds
        every multi-line command, so

            cat > /tmp/patch.py << 'EOF'
            <python body>
            EOF
            python3 /tmp/patch.py

        collapsed to an unterminated heredoc which wrote the literal text
        "python3 /tmp/patch.py" into the file and executed nothing, exiting 0
        with no output. Agent edits appeared to succeed while never landing.
        """
        cmd_stripped = cmd.strip()
        if not cmd_stripped:
            return ""
        exec_cmd = cmd_stripped if already_extracted else self.extract_action_command(cmd)
        if not exec_cmd or exec_cmd == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
            return "[Task Completed / Submitted]"
        if exec_cmd.startswith("[Harness:"):
            return exec_cmd

        try:
            res = subprocess.run(
                exec_cmd,
                shell=True,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                env=self._command_env(),
                timeout=120
            )
            stdout = res.stdout
            stderr = res.stderr
            output = stdout + (f"\nSTDERR:\n{stderr}" if stderr else "")
            return output if output else "[Command exited with 0]"
        except subprocess.TimeoutExpired:
            return "[Error: Command timed out after 120 seconds]"
        except Exception as e:
            return f"[Execution Error: {str(e)}]"

    # -- Arm 9: write-through viewport echo ----------------------------------

    def _viewport_echo(
        self,
        writes: List[str],
        before_contents: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate a viewport block for each file modified this step.

        Unlike Arms 1-8, which retroactively rewrite prompt history, this
        operates at the tool-output layer: the viewport is appended to the
        command's output *before* it enters the message history, making the
        transcript strictly append-only and preserving KV-cache prefixes.

        The mechanism itself lives in src/harness/viewport.py, decoupled from
        this scaffold and from mini-SWE-agent, so another framework's own
        scaffold can reuse it directly.
        """
        return self._viewport_blocks(writes, before_contents)[0]

    def _viewport_blocks(
        self,
        writes: List[str],
        before_contents: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, List[str]]:
        """The viewport text and the written files it shows; only those may be
        registered as seen (see viewport.viewport_blocks)."""
        if self.arm not in WRITE_THROUGH_ARMS or not writes:
            return "", []
        if self.arm in DIFF_ARMS:
            return tail_diff_blocks(self.workspace_dir, writes, self.freshness_hook, before_contents,
                                    current_step=self.step_counter)
        return viewport_blocks(self.workspace_dir, writes, self.freshness_hook, before_contents)

    def get_context_messages(self) -> List[Dict[str, Any]]:
        """Return conversation messages transformed according to active intervention arm."""
        return apply_intervention_arm(
            messages=self.messages,
            bindings=self.freshness_hook.bindings,
            arm=self.arm,
            max_tokens=self.max_tokens,
            current_tokens=sum(len(m.get("content", "")) // 4 for m in self.messages),
            workspace_root=self.workspace_dir,
        )

    def observe_step(
        self,
        command: str,
        output: str,
        msg_obs_idx: int,
        before: Dict[str, tuple],
        live_run: bool = True,
        parsed_reads: Optional[List[str]] = None,
        parsed_writes: Optional[List[str]] = None,
        referenced: Optional[List[str]] = None,
        step_id: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """Bind the files a step read and invalidate those it wrote.

        Shared by this scaffold and by the instrumented mini-swe-agent run, so
        that both score staleness with exactly one implementation.

        `before` is the workspace snapshot taken before the command ran; the
        filesystem diff, not the command text, decides which writes landed.
        """
        step_id = self.step_counter if step_id is None else step_id
        if parsed_reads is None or parsed_writes is None:
            _, p_reads, p_writes = self.parser.extract_bash_file_operations(command)
            parsed_reads = list(p_reads) if parsed_reads is None else parsed_reads
            parsed_writes = list(p_writes) if parsed_writes is None else parsed_writes
        if referenced is None:
            referenced = self.resolve_referenced_files(command)

        reads, writes = list(parsed_reads), list(parsed_writes)
        if live_run:
            observed_writes = self.diff_snapshots(before, self.snapshot_workspace())
            # Normalize to workspace-relative form so the same file referenced
            # absolutely and relatively is one binding, not two, and drop paths
            # outside the workspace (/tmp scratch files are not task scope).
            # Only writes the filesystem confirms count: a parsed write whose
            # target did not change (a sed pattern that matched nothing) is
            # not a write, and neither is a rewrite that left the bytes as
            # they were when the harness last hashed the file.
            writes = self._drop_unchanged(self._normalize_paths(observed_writes))
            # A file that is both shown and modified in one step is the
            # inspect-then-edit pattern the hazard depends on, so its read is
            # kept rather than discarded. Excluding it meant a step that
            # grepped a file and then patched it registered no observation, so
            # no binding existed for the write to invalidate and the hazard was
            # unobservable. Ordering below recovers which came first.
            reads = self._normalize_paths(reads + list(referenced))

        # Register in the order the step performed the operations. Command text
        # supplies the ordering; the filesystem snapshot decides which writes
        # actually landed, so a write parsed from text that changed nothing is
        # not registered.
        written_set = set(writes)
        registered_writes = set()
        ordered = self.ordered_file_events(command) if live_run else []

        for op, path in ordered:
            if op == "R":
                if path in reads:
                    self.freshness_hook.register_read(msg_obs_idx, step_id, path, output)
            elif path in written_set:
                self.freshness_hook.register_write(step_id, path, new_content=command)
                registered_writes.add(path)

        # Reads and writes the line-level pass could not attribute (heredoc
        # bodies, mocked steps) still need recording; order is unknown, so the
        # conservative inspect-then-edit order is used for those alone.
        for r_file in reads:
            if not any(op == "R" and p == r_file for op, p in ordered):
                self.freshness_hook.register_read(msg_obs_idx, step_id, r_file, output)
        for w_file in writes:
            if w_file not in registered_writes:
                self.freshness_hook.register_write(step_id, w_file, new_content=command)

        return reads, writes

    def step(self, action_cmd: str, mock_obs: Optional[str] = None) -> Dict[str, Any]:
        """Execute one step of the agent turn."""
        self.step_counter += 1

        # 1. Record assistant action, cut off after its one command so that
        # outputs the model imagined for itself never enter the context.
        raw_len = len(action_cmd or "")
        action_cmd = self.truncate_to_action(action_cmd or "")
        discarded_tail_chars = raw_len - len(action_cmd)
        self.messages.append({"role": "assistant", "content": action_cmd})
        msg_action_idx = len(self.messages) - 1

        # Heuristic extraction of reads and writes from extracted command
        clean_cmd = self.extract_action_command(action_cmd)
        _, reads, writes = self.parser.extract_bash_file_operations(clean_cmd or action_cmd)

        # Command text alone is an unreliable witness, so corroborate it
        # against the filesystem: candidate paths must exist to count as reads,
        # and writes are taken from an actual before/after diff.
        referenced = self.resolve_referenced_files(clean_cmd or action_cmd)
        live_run = mock_obs is None
        before = self.snapshot_workspace() if live_run else {}
        before_contents: Dict[str, str] = {}
        if live_run and self.arm in WRITE_THROUGH_ARMS:
            for r_file in referenced:
                abs_p = os.path.join(self.workspace_dir, r_file)
                if os.path.isfile(abs_p):
                    try:
                        with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                            before_contents[r_file] = f.read()
                    except OSError:
                        pass

        # 2. Execute command
        if mock_obs is not None:
            output = mock_obs
        else:
            output = self.execute_bash_command(clean_cmd, already_extracted=True)

        # 3. Record environment observation
        self.messages.append({"role": "user", "content": output})
        msg_obs_idx = len(self.messages) - 1

        # 4. Update freshness tracking
        reads, writes = self.observe_step(
            command=clean_cmd or action_cmd,
            output=output,
            msg_obs_idx=msg_obs_idx,
            before=before,
            live_run=live_run,
            parsed_reads=reads,
            parsed_writes=writes,
            referenced=referenced,
        )

        # 4b. Arm 9 (Write-Through Viewport): Append viewport echo and refresh binding
        if self.arm in WRITE_THROUGH_ARMS and writes:
            viewport, shown = self._viewport_blocks(writes, before_contents=before_contents)
            if viewport:
                output += viewport
                self.messages[msg_obs_idx]["content"] = output
                for w_file in shown:
                    self.freshness_hook.register_read(msg_obs_idx, self.step_counter, w_file, output)

        # 5. Apply intervention arm
        stale_bindings = self.freshness_hook.get_stale_bindings()
        transformed_messages = self.get_context_messages()

        return {
            "step_id": self.step_counter,
            "action": action_cmd,
            "observation": output,
            "invalid_observations_count": len(stale_bindings),
            "stale_bindings": [b.file_path for b in stale_bindings],
            "context_messages_count": len(transformed_messages),
            "terminated": self.signals_completion(action_cmd),
            # Recorded per step so a trace can be audited after the fact. Their
            # absence made it impossible to tell whether a zero invalidation
            # count meant the agent never wrote, or the write went unseen.
            "reads_detected": list(reads),
            "writes_detected": list(writes),
            # How much of the response followed the action and was dropped.
            # Large values mean the model was scripting its own session.
            "discarded_tail_chars": discarded_tail_chars,
        }
