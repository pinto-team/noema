# -*- coding: utf-8 -*-
"""
NOEMA • sleep package (V0)

Offline "sleep/consolidation" utilities in one place.

Quick start:
    from sleep import SleepCfg, run_sleep_cycle, run_once
    rep = run_once()                     # default settings
    # or:
    cfg = SleepCfg(episodes_root="data/episodes", build_tfidf=True)
    rep = run_sleep_cycle(cfg, verbose=True)
"""

from __future__ import annotations

from typing import Any, Dict

from .offline import SleepCfg, run_once, run_sleep_cycle

__all__ = [
    "SleepCfg",
    "run_sleep_cycle",
    "run_once",
]
