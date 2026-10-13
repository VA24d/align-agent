"""A live, Terminus2-style agent loop, for testing whether the write-through
viewport fix generalizes beyond mini-SWE-agent (Part C of the best-paper
revision plan; see docs/write-through-reference.md for the fix itself).

Terminus2 has no installable Python package with an Agent/Environment API to
subclass the way mini-SWE-agent has -- it is understood here only through the
published trajectory corpus src/parsers/terminus_parser.py already parses.
That corpus establishes the real mechanics this module reproduces:

  * per-turn model output is JSON, `{"analysis", "plan", "commands":
    [{"keystrokes", "duration"}], "task_complete"}` -- not mini-SWE-agent's
    single fenced-bash-block convention, so DefaultAgent's action parser
    (which hard-requires exactly one action per turn) cannot be reused here.
  * the agent drives a real interactive terminal with keystroke-level input,
    not one-shot subprocess calls: "the terminal's output arrives in the next
    episode's prompt and carries no exit status, so every write is taken to
    have landed" (terminus_parser.py). There is no discrete per-command
    observation message the way mini-SWE-agent has one message per action.

This module reproduces both properties faithfully -- real tmux inside the
task's own SWE-bench container, driven via `docker exec` (matching upstream
Terminal-Bench's own mechanism), and no exit code ever surfaced to the model
-- rather than approximating them with simpler subprocess calls, because the
paper's claim is specifically that the fix survives contact with a framework
built this way.

What *is* reused, verbatim, because it was already framework-agnostic:
`MiniSWEScaffold` as a pure instrumentation toolbox (staleness bookkeeping,
snapshot-diff write detection, shell-text parsing -- see its docstring),
`viewport_echo` (the fix itself), and `DockerEnvironment` from the installed
mini-swe-agent package for container lifecycle (it starts one persistent
container per task via `docker run -d ... sleep <timeout>` and is driven
by `docker exec`, so the same container_id this module's tmux commands
target is the one the rest of the harness already knows how to provision
and clean up).

Disclosure this run needs in the paper (Threats to Validity, matching the
house style already used there): Terminus2's own published 222-trajectory
corpus (feeding RQ1's Terminus2 row) is Terminal-Bench tasks, not SWE-bench
instances. This module runs Terminus2's real agent loop against SWE-bench
Verified instead, for comparability with the paper's other live RQ2 arms.
The two should not be conflated or pooled.
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from minisweagent.environments.docker import DockerEnvironment

from src.metrics.corruption_oracles import check_ast_parseability
from src.parsers.terminus_parser import TerminusParser

from .arms import InterventionArm
from .completion import post_completion
from .miniswe_scaffold import MiniSWEScaffold
from .viewport import viewport_blocks

logger = logging.getLogger("terminus_live")

HARNESS_PROTOCOL = "episodic-terminus2-live-v1/freshness-v13"
CONTAINER_CWD = "/testbed"
DOCKER_BIN = os.environ.get(
    "SGA_DOCKER",
    "/usr/local/bin/docker" if os.path.exists("/usr/local/bin/docker")
    else "/Applications/Docker.app/Contents/Resources/bin/docker",
)
TMUX_SESSION = "main"

# Same sampling preset the existing Qwen pilot uses (src/harness/mini_real.py
# LMStudioModel), so a second-framework comparison isn't also a sampling
# comparison. Duplicated here rather than imported: this module is
# deliberately Terminus2-specific glue, unlike viewport.py, so importing
# mini_real's model classes (built around DefaultAgent's single-action
# contract) would pull in machinery this loop doesn't use.
QWEN_CODING_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0.0,
}

TERMINUS_SYSTEM_TEMPLATE = """You are an autonomous coding agent working in a real Linux terminal to resolve a software engineering task.

Each turn, respond with exactly one JSON object and nothing else -- no markdown fences, no prose outside the object:

{{
  "analysis": "<brief analysis of the current terminal state>",
  "plan": "<what you are about to do and why>",
  "commands": [
    {{"keystrokes": "<a single shell command, no trailing newline>", "duration": <seconds to wait before checking output, a float>}}
  ],
  "task_complete": <true only once the fix is applied and you are done -- do not submit a patch, the workspace itself is graded>
}}

You may include more than one command in "commands"; they run in the order given. You will see their combined terminal output next turn, with no indication of which command succeeded or failed -- the terminal gives you no exit status. Use short durations (1-3 seconds) for fast commands; for a command that may take longer, either give it a longer duration or send a short-duration command and poll with an empty keystroke ("keystrokes": "", "duration": 10.0) on a later turn rather than blocking a long time up front. Never wait longer than 60 seconds in one command.

The workspace is the answer: do not try to submit a diff or patch file. When you set "task_complete": true, the run ends and the workspace's current state is graded as-is.

Task:
{task}
"""

FORMAT_RETRY_MESSAGE = (
    'Your last response was not a single valid JSON object matching the required schema. '
    'Respond with exactly one JSON object: {"analysis": ..., "plan": ..., "commands": '
    '[{"keystrokes": ..., "duration": ...}], "task_complete": ...}. No other text.'
)


# --- tmux driver, via docker exec against an already-running container ----

def _docker_exec(container_id: str, args: List[str], docker_bin: str = DOCKER_BIN,
                  timeout: float = 15.0) -> subprocess.CompletedProcess:
    cmd = [docker_bin, "exec", container_id, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def tmux_available(container_id: str, docker_bin: str = DOCKER_BIN) -> bool:
    try:
        r = _docker_exec(container_id, ["which", "tmux"], docker_bin, timeout=10)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def install_tmux(container_id: str, docker_bin: str = DOCKER_BIN, timeout: float = 240.0) -> bool:
    """Best-effort `apt-get install tmux`. Returns whether tmux is available afterward.

    `apt-get update` dominates the cost, not the package itself: an empty
    image apt cache means a full package-index fetch, timed at ~141s against
    a real SWE-bench astropy image on this network path (2m21s wall time for
    the whole update+install). 90s was an earlier, untested guess and timed
    out on both smoke-test tasks before tmux ever finished installing.
    """
    try:
        _docker_exec(
            container_id,
            ["bash", "-lc", "apt-get update -qq && apt-get install -y -qq tmux"],
            docker_bin, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("tmux install timed out in container %s", container_id)
    return tmux_available(container_id, docker_bin)


def tmux_new_session(container_id: str, docker_bin: str = DOCKER_BIN,
                      session: str = TMUX_SESSION, cols: int = 200, rows: int = 50) -> None:
    r = _docker_exec(
        container_id,
        ["tmux", "new-session", "-d", "-x", str(cols), "-y", str(rows), "-s", session, "-c", CONTAINER_CWD],
        docker_bin, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"tmux new-session failed in container {container_id}: {r.stderr.strip()}")


def tmux_send_keys(container_id: str, keystrokes: str, docker_bin: str = DOCKER_BIN,
                    session: str = TMUX_SESSION) -> None:
    """Send literal keystrokes, then Enter, as two separate calls.

    `-l` sends the text literally rather than reinterpreting it as tmux key
    names -- without it, text containing sequences that look like key names
    (e.g. a literal "C-c" substring in a string the model is echoing) would be
    silently misinterpreted rather than typed. Enter is sent unliteral on
    purpose, as an actual keypress.
    """
    if keystrokes:
        _docker_exec(container_id, ["tmux", "send-keys", "-l", "-t", session, keystrokes], docker_bin, timeout=15)
    _docker_exec(container_id, ["tmux", "send-keys", "-t", session, "Enter"], docker_bin, timeout=15)


def tmux_capture(container_id: str, docker_bin: str = DOCKER_BIN, session: str = TMUX_SESSION) -> str:
    """The visible pane, not the session's full scrollback.

    An earlier version passed `-S -` (capture from the start of history) so a
    long command's output wouldn't be truncated to whatever last fit on
    screen. That was wrong: `-S -` captures everything since the tmux session
    itself started, not since the last capture, so every episode's "New
    Terminal Output" re-included the entire transcript so far on top of
    itself. Confirmed empirically against a live smoke-test trace -- observed
    message size grew from 253 to 15,144 characters over 18 episodes with no
    corresponding growth in actual new output -- and it was the dominant
    cause of exhausting the model's 32K context by episode ~18-20, well
    before any episode budget was reached. Bounding to the visible pane is
    also the more faithful choice: it matches what a person actually sees in
    a real terminal after a long-output command scrolls past, which is what
    Terminus2's real mechanism is modeling in the first place.
    """
    r = _docker_exec(container_id, ["tmux", "capture-pane", "-p", "-t", session], docker_bin, timeout=15)
    return r.stdout


def capture_after_commands(container_id: str, docker_bin: str = DOCKER_BIN, session: str = TMUX_SESSION) -> str:
    """The episode's observation: whatever is on screen once the caller has
    already slept for the commands' own stated `duration`s.

    No injected marker command here on purpose -- an earlier version sent
    `echo <uuid>` and polled for it to settle, but that types a command the
    model never issued into whatever currently has the terminal's input
    focus. If a prior command left something other than a shell prompt in
    the foreground (an editor, a REPL, a long job with no background/poll
    split), the injected echo lands as *input to that program* instead of
    executing, which is both a fidelity violation (the harness now acts
    without the model's say-so) and unreliable (many such programs disable
    local echo, so the marker may not even become visible to poll for).
    Terminus2 itself has no exit-status signal and relies on the model's own
    stated `duration` plus explicit poll turns (an empty-keystrokes command
    with its own `duration`) to decide when to look again -- so the harness
    doesn't need to invent its own settle detection on top of that; it only
    needs to report what's on screen after waiting as long as it was told to.
    """
    return tmux_capture(container_id, docker_bin, session)


# --- model backend: raw completion, no action-regex parsing --------------

def _complete(messages: List[Dict[str, str]], model_name: str, api_base: str,
               api_key: str = "lm-studio", timeout: int = 360,
               extra: Optional[Dict[str, Any]] = None) -> Tuple[str, str, Dict[str, Any]]:
    """One chat-completion call. Returns (content, finish_reason, usage).

    finish_reason lets the caller tell a truncated response ("length") apart
    from one the model actually finished but got wrong -- otherwise both look
    identical (a JSON parse failure) and a truncation-heavy run would read as
    a schema-compliance problem with Terminus2's stricter format instead of
    what it actually is, an output-budget problem.

    usage is returned, not just logged, because scripts/recompute_metrics.py
    already sums `message["extra"]["usage"]["total_tokens"]` across a trace's
    messages to compute the paper's token-cost column (see mini_real.py's
    identical pattern) -- the caller only needs to attach it to the assistant
    message it appends, not duplicate that summation.
    """
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {"model": model_name, "messages": messages, **(extra or {})}
    headers = {"Content-Type": "application/json"}
    if api_key and api_key not in ("lm-studio", "ollama"):
        headers["Authorization"] = f"Bearer {api_key}"
    response = post_completion(url, payload, headers, timeout)
    choice = response["choices"][0]
    content = choice.get("message", {}).get("content") or ""
    finish_reason = choice.get("finish_reason") or ""
    usage = response.get("usage") or {}
    return content, finish_reason, usage


def _strip_reasoning(text: str) -> str:
    """Drop a thinking model's chain-of-thought preamble, if present.

    Qwen 3.6 A3B is a thinking model: against this vLLM server it does not
    separate reasoning into its own field the way `reasoning_content` would --
    the whole chain of thought lands directly in `content`, ending in a bare
    `</think>` with no matching opening tag (confirmed against real smoke-test
    traces: every turn's content began with free prose and only ever closed
    the tag, never opened it), followed by the actual JSON object.
    """
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    return text


def _parse_episode_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse one turn's JSON response, tolerating a stray code fence and a
    reasoning preamble (see `_strip_reasoning`) rather than treating every
    such turn as a format failure."""
    text = _strip_reasoning(raw)
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Last resort: the model wrapped valid JSON in extra commentary this
        # didn't anticipate. Try the substring between the first '{' and the
        # matching last '}' before giving up.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(parsed, dict) or "commands" not in parsed:
        return None
    if not isinstance(parsed.get("commands"), list):
        return None
    return parsed


# --- the episode loop -------------------------------------------------------

class TerminusLiveAgent:
    """Drives one real tmux session, inside one task's SWE-bench container,
    for the duration of a live Terminus2-style run.

    Purely an orchestrator: staleness bookkeeping and the write-through fix
    itself live in `self.instr` (a MiniSWEScaffold used only as an
    instrumentation toolbox, never as an agent loop -- see its docstring and
    mini_real.py's FreshnessAgent for the same pattern applied to
    mini-SWE-agent).
    """

    def __init__(
        self,
        *,
        arm: InterventionArm,
        workspace_dir: str,
        model_name: str,
        api_base: str,
        api_key: str = "lm-studio",
        image: Optional[str] = None,
        episode_limit: int = 40,
        wall_time_limit_seconds: int = 1800,
        command_timeout: int = 120,
        container_cpus: str = "4",
        container_memory: str = "8g",
        use_mtime_cache: bool = True,
        docker_bin: str = DOCKER_BIN,
        max_tokens: int = 4096,
    ):
        self.arm = arm
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.episode_limit = episode_limit
        self.wall_time_limit_seconds = wall_time_limit_seconds
        self.docker_bin = docker_bin
        self.max_tokens = max_tokens
        self.n_format_errors_truncated = 0

        if not image:
            raise ValueError("TerminusLiveAgent requires a real SWE-bench image: no local/pty fallback is wired up yet.")

        self.env = DockerEnvironment(
            image=image,
            cwd=CONTAINER_CWD,
            timeout=command_timeout,
            executable=self.docker_bin,
            run_args=[
                "--rm",
                "--platform", "linux/amd64",
                f"--cpus={container_cpus}",
                f"--memory={container_memory}",
                "-v", f"{self.workspace_dir}:{CONTAINER_CWD}",
            ],
        )
        # Instrumentation toolbox only -- never an agent loop here. Parser is
        # TerminusParser for provenance honesty; extract_bash_file_operations
        # is inherited from BaseTrajectoryParser and behaves identically to
        # MiniSWEParser's for shell-text commands either way.
        self.instr = MiniSWEScaffold(
            arm=arm,
            workspace_dir=self.workspace_dir,
            use_mtime_cache=use_mtime_cache,
            parser_cls=TerminusParser,
        )

        self.messages: List[Dict[str, str]] = []
        self.step_records: List[Dict[str, Any]] = []
        self.n_format_errors = 0
        self.n_calls = 0
        self.driver = "tmux"
        # Rolling cache of "what did this path look like the last time we
        # observed it", updated after every episode for every read/written
        # file -- not just paths resolved from the current episode's command
        # text, so a write made via an editor session spanning multiple
        # episodes still has a valid diff anchor for the viewport. A write
        # still sitting in an unsaved editor buffer is not observable this
        # way; that's a real, disclosed limitation, not a bug.
        self.before_contents_cache: Dict[str, str] = {}
        self._tmux_ready = False

    def _ensure_tmux(self) -> None:
        if self._tmux_ready:
            return
        if not tmux_available(self.env.container_id, self.docker_bin):
            ok = install_tmux(self.env.container_id, self.docker_bin)
            if not ok:
                raise RuntimeError(
                    f"tmux is not available and could not be installed in container "
                    f"{self.env.container_id}; no pty fallback is wired up yet."
                )
        tmux_new_session(self.env.container_id, self.docker_bin)  # raises on failure
        # A session that exists but never produces a prompt (a shell that
        # crashed on start, or a session tmux silently didn't attach a pane
        # to) would otherwise surface only much later, as 40 episodes of
        # blank observations indistinguishable from a clean, uneventful task.
        # `new-session -d` returns before the spawned shell has necessarily
        # rendered its first prompt into the pane buffer -- confirmed
        # empirically (capture-pane immediately after new-session came back
        # blank on a real SWE-bench container even though the shell was
        # already running, per `ps aux`; capturing again after a moment saw
        # the prompt). This re-observes only -- no keystrokes sent -- so it
        # does not repeat the fidelity problem the removed marker-based
        # settle detector had.
        first_capture = ""
        for _ in range(6):
            first_capture = tmux_capture(self.env.container_id, self.docker_bin)
            if first_capture.strip():
                break
            time.sleep(0.5)
        if not first_capture.strip():
            raise RuntimeError(
                f"tmux session started in container {self.env.container_id} but its pane is still empty "
                f"after 3s; the shell inside it may not have come up."
            )
        self._tmux_ready = True

    def close(self) -> None:
        """Tear down the container. Call this only after everything that
        needs the live workspace (git_changed_files, blast-radius scoring)
        has already run -- see commit 9348edd, the same cleanup-before-scoring
        ordering bug already found and fixed once for mini-SWE-agent."""
        self.env.cleanup()

    def _syntax_error_in(self, writes: List[str]) -> bool:
        for rel in writes:
            if not rel.endswith(".py"):
                continue
            path = rel if os.path.isabs(rel) else os.path.join(self.workspace_dir, rel)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                continue
            if not check_ast_parseability(content, rel).is_valid:
                return True
        return False

    def _update_before_contents(self, paths: List[str]) -> None:
        for rel in paths:
            abs_path = os.path.join(self.workspace_dir, rel)
            if not os.path.isfile(abs_path):
                continue
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    self.before_contents_cache[rel] = f.read()
            except OSError:
                continue

    def run(self, task: str) -> Dict[str, Any]:
        """Run to task_complete, an episode limit, a wall-time limit, or a
        repeated-format-error abort. Returns exit info; step_records/messages
        are populated on self for the caller to serialize into a trace."""
        self._ensure_tmux()
        started = time.time()

        initial_capture = tmux_capture(self.env.container_id, self.docker_bin)
        self.messages = [
            {"role": "system", "content": TERMINUS_SYSTEM_TEMPLATE.format(task=task)},
            {"role": "user", "content": f"Current Terminal Screen:\n\n{initial_capture}"},
        ]

        exit_status = "LimitsExceeded"
        episode = 0
        consecutive_format_errors = 0

        while episode < self.episode_limit:
            if time.time() - started > self.wall_time_limit_seconds:
                exit_status = "TimeExceeded"
                break

            try:
                raw, finish_reason, usage = _complete(
                    self.messages, self.model_name, self.api_base, self.api_key,
                    extra={**QWEN_CODING_SAMPLING, "max_tokens": self.max_tokens},
                )
            except Exception as exc:
                logger.error("Completion failed at episode %d: %r", episode, exc)
                exit_status = "RuntimeError"
                break
            self.n_calls += 1

            parsed = _parse_episode_response(raw)
            if parsed is None:
                consecutive_format_errors += 1
                self.n_format_errors += 1
                if finish_reason in ("length", "max_tokens"):
                    self.n_format_errors_truncated += 1
                # Reasoning stripped before it enters history, same as the
                # success branch below -- see that branch's comment for why.
                self.messages.append(
                    {"role": "assistant", "content": _strip_reasoning(raw), "extra": {"usage": usage}}
                )
                if consecutive_format_errors >= 3:
                    exit_status = "RepeatedFormatError"
                    break
                self.messages.append({"role": "user", "content": FORMAT_RETRY_MESSAGE})
                continue
            consecutive_format_errors = 0
            # Stored as the re-serialized parse, not the raw response: a
            # thinking model's chain-of-thought is not meant to be replayed
            # back into its own context on the next turn (that's how every
            # reasoning-model integration treats it, and it isn't optional
            # here -- feeding the full raw text back grew the transcript by
            # ~7K tokens/turn against this model and blew the 32K context
            # budget by episode 4-6 in the smoke test). Re-serializing the
            # already-parsed object, rather than just stripping the raw text,
            # also drops any leftover chatter the brace-matching fallback
            # tolerated, so what's stored is always clean, minimal JSON.
            self.messages.append(
                {"role": "assistant", "content": json.dumps(parsed), "extra": {"usage": usage}}
            )

            episode += 1
            commands = parsed.get("commands") or []
            keystrokes_joined = "\n".join(
                str(c.get("keystrokes", "")).rstrip("\n") for c in commands if isinstance(c, dict)
            )

            referenced = self.instr.resolve_referenced_files(keystrokes_joined) if keystrokes_joined else []
            missing_before = [p for p in referenced if p not in self.before_contents_cache]
            self._update_before_contents(missing_before)

            before_snapshot = self.instr.snapshot_workspace()
            for c in commands:
                if not isinstance(c, dict):
                    continue
                ks = str(c.get("keystrokes", ""))
                if ks:
                    tmux_send_keys(self.env.container_id, ks, self.docker_bin)
                try:
                    duration = float(c.get("duration") or 1.0)
                except (TypeError, ValueError):
                    duration = 1.0
                time.sleep(min(max(duration, 0.0), 60.0))

            capture = capture_after_commands(self.env.container_id, self.docker_bin)

            msg_obs_idx = len(self.messages)  # index the observation message will land at, below
            reads, writes = self.instr.observe_step(
                command=keystrokes_joined,
                output=capture,
                msg_obs_idx=msg_obs_idx,
                before=before_snapshot,
                live_run=True,
                step_id=episode,
            )

            syntax_error = self._syntax_error_in(writes)

            viewport_text = ""
            if self.arm == InterventionArm.ARM_9_WRITE_THROUGH and writes:
                viewport_text, shown = viewport_blocks(
                    self.workspace_dir, writes, self.instr.freshness_hook, self.before_contents_cache,
                )
                if viewport_text:
                    obs_with_viewport = capture + viewport_text
                    for w_file in shown:
                        self.instr.freshness_hook.register_read(msg_obs_idx, episode, w_file, obs_with_viewport)

            self._update_before_contents(list(set(reads) | set(writes)))

            stale = self.instr.freshness_hook.get_stale_bindings()
            self.messages.append({"role": "user", "content": f"New Terminal Output:\n\n{capture}{viewport_text}"})

            self.step_records.append({
                "step_id": episode,
                "command": keystrokes_joined,
                "commands_per_episode": len(commands),
                # capture + viewport_text, matching exactly what the model's
                # own next-turn message contained (see the messages.append
                # just above) -- stored separately here rather than derived,
                # so the trace's own observation field cannot drift from what
                # the model actually saw. Caught live: recompute_metrics.py's
                # compute_coherence() reads this field, and it was recording
                # the bare capture only, silently hiding the write-through
                # arm's own intervention from the exact analysis meant to
                # measure its effect.
                "observation": capture + viewport_text,
                "returncode": None,  # Terminus2 has none -- never invent one.
                "reads_detected": reads,
                "writes_detected": writes,
                "invalid_observations_count": len(stale),
                "stale_bindings": [b.file_path for b in stale],
                "syntax_error": syntax_error,
                "finish_reason": finish_reason,
            })

            if parsed.get("task_complete") is True:
                exit_status = "Submitted"
                break
        else:
            exit_status = "LimitsExceeded"

        return {
            "exit_status": exit_status,
            "total_episodes": episode,
            "wall_seconds": round(time.time() - started, 1),
            "n_calls": self.n_calls,
            "n_format_errors": self.n_format_errors,
            "n_format_errors_truncated": self.n_format_errors_truncated,
        }
