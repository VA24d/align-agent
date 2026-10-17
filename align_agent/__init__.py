"""ALIGN: find edits a coding agent composed against a stale view of a file.

`align-agent analyze` replays agent trajectories (mini-SWE-agent, SWE-agent,
OpenHands, Terminus 2, Harbor/ATIF, OpenAI-style tool calls) and the session
logs of Claude Code and Codex CLI, and reports every anchored edit made to a
file whose last view the agent's own writes had outdated, with the step that
held the view, how the view was delivered, and whether the edit went wrong
because of it.
"""

__version__ = "0.4.4"
