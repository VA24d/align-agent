"""Parsers package for heterogeneous agent trajectories."""

from .base_parser import BaseTrajectoryParser, TrajectoryStep, ActionKind
from .miniswe_parser import MiniSWEParser
from .sweagent_parser import SWEAgentParser
from .openhands_parser import OpenHandsParser
from .terminus_parser import TerminusParser

__all__ = [
    "BaseTrajectoryParser",
    "TrajectoryStep",
    "ActionKind",
    "MiniSWEParser",
    "SWEAgentParser",
    "OpenHandsParser",
    "TerminusParser",
    "get_parser_for_agent",
]


def get_parser_for_agent(agent_name: str) -> BaseTrajectoryParser:
    """Factory function returning the appropriate parser for an agent framework."""
    if agent_name == "mini-SWE-agent":
        return MiniSWEParser()
    elif agent_name == "SWE-agent":
        return SWEAgentParser()
    elif agent_name == "OpenHands":
        return OpenHandsParser()
    elif agent_name == "Terminus2":
        return TerminusParser()
    else:
        # Default fallback to mini-SWE parser
        return MiniSWEParser()
