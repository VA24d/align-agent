"""Metrics package for staleness replay and statistical analysis.

The statistics helpers pull in statsmodels, scipy and numpy. Importing them
here made every consumer of this package pay that cost, including the
experiment runner, which needs nothing but the corruption oracles: on one
machine a scipy/numpy version mismatch in the active interpreter turned
`from wt_src.metrics.corruption_oracles import ...` into an import error, which
has nothing to do with what the caller asked for. They are resolved on first
attribute access instead.

The paper's numbers come from src/metrics/replay.py (one stale-edit
definition for both research questions) via scripts/run_phase1_analysis.py
and scripts/recompute_metrics.py.
"""

from typing import Any

_LAZY = {
    "compute_windowed_odds_ratios": ".statistics",
    "fit_length_controlled_logistic_regression": ".statistics",
    "fit_stale_edit_association": ".statistics",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)


__all__ = [
    "compute_windowed_odds_ratios",
    "fit_length_controlled_logistic_regression",
    "fit_stale_edit_association",
]
