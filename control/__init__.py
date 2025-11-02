# -*- coding: utf-8 -*-
"""
NOEMA • control package (V0)

Public surface
--------------
- generate_candidates(state, plan, wm=None, tool_registry=None) -> List[Action]
- decide(state, candidates, predict_fn, *, r_ext=0.0, ...) -> (Action, ranked_details)
- plan_and_decide(state, plan, generate_candidates_fn, predict_fn, *, ...) -> (Action, rationale)
- score_candidate(state, action, predict_fn, *, ...) -> (score, details)

Config
------
- RewardSpec, get_default_spec()

Notes
-----
- This package is lightweight and dependency-free.
- World/Value modules are expected to be available (see type hints).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

# Re-export the main entry points
from .candidates import generate as generate_candidates
from .policy import decide, score_candidate, RewardSpec, get_default_spec
from .planner import plan_and_decide

if TYPE_CHECKING:
    # Only for type hints; not imported at runtime
    from world import State, Action, Latent  # noqa: F401

__all__ = [
    "generate_candidates",
    "decide",
    "plan_and_decide",
    "score_candidate",
    "RewardSpec",
    "get_default_spec",
]

__version__ = "0.1.0"
