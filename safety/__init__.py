# -*- coding: utf-8 -*-
"""
NOEMA • safety package (V0)

Exports a minimal DSL for safety rules and a runtime shield.
"""

from __future__ import annotations

from .dsl import (
    load_policies as load_rules,
    evaluate,
    safe_decide_allow,
    Rule,
    Decision,
)

from .shield import (
    check_action,
    enforce,
    gate_candidates,
    reason_text,
    check,          # adapter for main.py (state, action) -> (allow, patch, meta)
)

__all__ = [
    # DSL
    "load_rules",
    "evaluate",
    "safe_decide_allow",
    "Rule",
    "Decision",
    # Shield
    "check_action",
    "enforce",
    "gate_candidates",
    "reason_text",
    "check",
]

if __name__ == "__main__":
    rules = load_rules()  # may be empty
    print(f"safety loaded — {len(rules)} rule(s)")
