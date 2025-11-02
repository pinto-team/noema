# -*- coding: utf-8 -*-
"""
NOEMA • safety/shield.py — Runtime safety shield (V0)

Responsibilities
----------------
- Load safety rules (from YAML/JSON if present).
- Decide for an action: allow / review / block.
- Provide both a high-level API (check_action/enforce/gate_candidates) and
  a minimal adapter `check(state, action)` required by main.py.

Exports
-------
load_rules(path="config/safety.yaml") -> List[Rule]
check_action(text, plan, action, state, rules, risk_base=0.0) -> Decision
enforce(text, plan, action, state, rules, on_block="clarify") -> (Action, Decision)
gate_candidates(text, plan, candidates, state, rules) -> (filtered, decisions)
check(state, action) -> (allow: bool, patch: dict, meta: dict)  # minimal adapter for main.py
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

# World action type (fallback-friendly)
try:
    from world import Action  # type: ignore
except Exception:
    from dataclasses import dataclass, field
    @dataclass
    class Action:
        kind: str
        name: str
        args: Dict[str, Any] = field(default_factory=dict)

from .dsl import load_policies, evaluate, Rule, Decision

# ------------------------- Rules loading & cache -------------------------

_RULES_CACHE: Optional[List[Rule]] = None

def load_rules(path: str = "config/safety.yaml") -> List[Rule]:
    """Load rules from YAML/JSON. Returns empty list if the file is missing."""
    global _RULES_CACHE
    _RULES_CACHE = load_policies(path)
    return _RULES_CACHE

def _ensure_rules() -> List[Rule]:
    global _RULES_CACHE
    if _RULES_CACHE is None:
        _RULES_CACHE = load_policies("config/safety.yaml")
    return _RULES_CACHE

# ------------------------------ Core checks ------------------------------

def _make_context(
    text: str,
    plan: Dict[str, Any],
    action: Action,
    state: Optional[Dict[str, Any]] = None,
    risk_base: float = 0.0,
) -> Dict[str, Any]:
    return {
        "text": text or "",
        "plan": dict(plan or {}),
        "action": {"name": action.name, "kind": action.kind, "args": dict(action.args or {})},
        "state": dict(state or {}),
        "risk_base": float(risk_base or 0.0),
    }

def check_action(
    text: str,
    plan: Dict[str, Any],
    action: Action,
    state: Optional[Dict[str, Any]],
    rules: List[Rule],
    *,
    risk_base: float = 0.0,
) -> Decision:
    """Return Decision(effect, matched, risk, reasons, details)."""
    ctx = _make_context(text, plan, action, state, risk_base=risk_base)
    return evaluate(ctx, rules)

# ------------------------------ Enforcement ------------------------------

def enforce(
    text: str,
    plan: Dict[str, Any],
    action: Action,
    state: Optional[Dict[str, Any]],
    rules: List[Rule],
    *,
    on_block: str = "clarify",     # "clarify" | "drop"
    risk_base: float = 0.0,
) -> Tuple[Action, Decision]:
    """
    Apply decision:
      - allow  → returns the same action
      - review → returns a safe clarification action (ask_clarify)
      - block  → returns ask_clarify (or drop behaviour)
    """
    dec = check_action(text, plan, action, state, rules, risk_base=risk_base)
    if dec.effect == "allow":
        return action, dec

    if on_block == "clarify":
        hint = "short" if (plan or {}).get("intent", "unknown") == "unknown" else "detail"
        alt = Action(kind="policy", name="ask_clarify", args={"hint": hint, "reason": ", ".join(dec.matched)})
        return alt, dec

    # drop → still return a safe clarify to keep pipeline stable
    alt = Action(kind="policy", name="ask_clarify", args={"reason": ", ".join(dec.matched)})
    return alt, dec

# --------------------------- Candidate gating ---------------------------

def gate_candidates(
    text: str,
    plan: Dict[str, Any],
    candidates: List[Action],
    state: Optional[Dict[str, Any]],
    rules: List[Rule],
    *,
    risk_base: float = 0.0,
) -> Tuple[List[Action], List[Tuple[Action, Decision]]]:
    """Filter candidates to safe ones; if none, add a safe clarify."""
    decisions: List[Tuple[Action, Decision]] = []
    safe_list: List[Action] = []

    for a in candidates:
        dec = check_action(text, plan, a, state, rules, risk_base=risk_base)
        decisions.append((a, dec))
        if dec.effect == "allow":
            safe_list.append(a)

    if not safe_list:
        safe_list = [Action(kind="policy", name="ask_clarify", args={"reason": "no_safe_candidate"})]

    # deduplicate
    out: List[Action] = []
    seen = set()
    for a in safe_list:
        key = (a.kind, a.name, tuple(sorted((a.args or {}).items())))
        if key not in seen:
            out.append(a)
            seen.add(key)

    return out, decisions

# --------------------------- Summary helpers ---------------------------

def reason_text(dec: Decision) -> str:
    """Compact human-readable summary of the decision."""
    prefix = {"allow": "✅", "review": "⚠️", "block": "⛔️"}.get(dec.effect, "")
    msg = " / ".join(dec.reasons) if dec.reasons else dec.effect
    return f"{prefix} {msg} [risk={dec.risk:.2f}, matched={','.join(dec.matched) or '-'}]"

# ---------------------- Minimal adapter for main.py ----------------------

def check(state: Any, action: Any) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    """
    Adapter expected by main.py:
    Input:  (state, action)
    Output: (allow: bool, patch: dict, meta: dict)

    - Uses loaded rules (lazy) with minimal context (no text/plan here).
    - Does not modify the action (patch = {}), just decides allow or not.
    """
    rules = _ensure_rules()

    # Normalize inputs to our Action type
    if not isinstance(action, Action):
        try:
            a = Action(kind=str(action.kind), name=str(action.name), args=dict(getattr(action, "args", {}) or {}))
        except Exception:
            a = Action(kind="policy", name="ask_clarify", args={})
    else:
        a = action

    st = {}
    try:
        st = {
            "u": float(getattr(state, "u", 0.0) if state is not None else 0.0),
            "conf": float(getattr(state, "conf", 0.0) if state is not None else 0.0),
        }
    except Exception:
        st = {}

    dec = check_action(text="", plan={"intent": "unknown"}, action=a, state=st, rules=rules, risk_base=0.0)
    allow = (dec.effect == "allow")
    patch: Dict[str, Any] = {}
    meta = {
        "effect": dec.effect,
        "matched": dec.matched,
        "risk": dec.risk,
        "reasons": dec.reasons,
        "n_rules": dec.details.get("n_rules", 0),
        "n_matched": dec.details.get("n_matched", 0),
    }
    return allow, patch, meta

if __name__ == "__main__":
    # Quick smoke test
    load_rules()  # from config/safety.yaml if present (else empty)
    a1 = Action(kind="tool", name="invoke_calc", args={"expr": "7*(5-2)"})
    a2 = Action(kind="tool", name="run_shell", args={"cmd": "rm -rf /"})
    for a in (a1, a2):
        allow, _, meta = check({"u": 0.2, "conf": 0.8}, a)  # type: ignore
        print(a.name, "→ allow?", allow, meta)
