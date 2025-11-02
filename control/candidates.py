# -*- coding: utf-8 -*-
"""
NOEMA • control/candidates.py — action candidate generation (V0)

Goal
-----
Given the current plan/intent + optional working memory context and an
optional tool registry, propose a small set of plausible Actions. This
module is *not* a decision maker; it only enumerates good options.

Inputs
------
- state : world.State (can use u/conf if needed)
- plan  : e.g. {"intent":"compute","args":{"expr":"2+2","raw":"..."}}
- wm    : (optional) memory.WorkingMemory for recent context
- tool_registry : (optional) to suggest safe generic tools

Output
------
- List[Action] (de-duplicated; args filled when possible)

Heuristics
----------
- unknown intent → always include ask_clarify (+ safe basics)
- compute intent → ensure a valid 'expr' (extract from raw or WM if missing)
- greeting intent → include reply_greeting
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

# world types (fallback stubs)
try:
    from world import State, Action  # type: ignore
except Exception:
    @dataclass
    class State:
        s: List[float]; u: float = 0.0; conf: float = 0.0
    @dataclass
    class Action:
        kind: str; name: str; args: Dict[str, Any]

# light normalization for math extraction
# (kept local to avoid heavy deps; aligns with lang/parse.py behavior)
_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_MATH_SYMS  = {"×": "*", "÷": "/", "−": "-", "–": "-", "—": "-"}

_SAFE_DEFAULT_TOOLS = [
    ("tool", "invoke_calc", {}),     # safe calculator
    ("policy", "ask_clarify", {}),   # clarifying question
]

_NUM_EXPR_RE = re.compile(r"[0-9+\-*/() \t]{2,}")

def _to_ascii_math(text: str) -> str:
    if not text:
        return ""
    t = text.translate(_FA_DIGITS).translate(_AR_DIGITS)
    for k, v in _MATH_SYMS.items():
        t = t.replace(k, v)
    return t

def _dedup(actions: List[Action]) -> List[Action]:
    seen = set()
    out: List[Action] = []
    for a in actions:
        key = (a.kind, a.name, tuple(sorted((a.args or {}).items())))
        if key not in seen:
            out.append(a)
            seen.add(key)
    return out

def _maybe_extract_expr(text: str) -> Optional[str]:
    t = _to_ascii_math(text or "")
    m = _NUM_EXPR_RE.search(t)
    if not m:
        return None
    expr = (m.group(0) or "").strip()
    if not expr:
        return None
    # guard: only the allowed charset
    return expr if re.fullmatch(r"[0-9+\-*/() \t]+", expr) else None

def _context_tail_text(wm, k: int = 3) -> str:
    if wm is None:
        return ""
    pairs = wm.context(k=k)
    return pairs[-1][0] if pairs else ""

# ----------------------------- API -----------------------------

def generate(
    state: State,
    plan: Dict[str, Any],
    wm: Optional[Any] = None,
    tool_registry: Optional[Any] = None,
) -> List[Action]:
    """
    Produce action candidates from intent + light context signals.
    """
    intent = (plan or {}).get("intent", "unknown")
    args   = dict((plan or {}).get("args", {}))

    cands: List[Action] = []

    # 1) greeting
    if intent == "greeting":
        cands.append(Action(kind="skill", name="reply_greeting", args={}))

    # 2) compute
    elif intent == "compute":
        expr = args.get("expr")
        if not isinstance(expr, str) or not expr.strip():
            raw = args.get("raw", "") or _context_tail_text(wm, k=3)
            expr = _maybe_extract_expr(raw) or "2+2"  # safe fallback
        cands.append(Action(kind="tool", name="invoke_calc", args={"expr": expr}))
        # if uncertainty is high, clarify first might help
        if float(getattr(state, "u", 0.0)) >= 0.5:
            cands.append(Action(kind="policy", name="ask_clarify", args={}))

    # 3) unknown/other intents → clarify + safe tools
    else:
        cands.append(Action(kind="policy", name="ask_clarify", args={}))
        if tool_registry and hasattr(tool_registry, "list_safe_basics"):
            try:
                for tool_name in tool_registry.list_safe_basics():
                    cands.append(Action(kind="tool", name=tool_name, args={}))
            except Exception:
                pass
        else:
            for kind, name, a in _SAFE_DEFAULT_TOOLS:
                cands.append(Action(kind=kind, name=name, args=a))

    # 4) small WM-based tweak: if last action was clarify & conf is decent,
    #    prefer the short clarify variant to avoid loops
    if wm is not None and len(wm) > 0:
        last = wm.last_action() or {}
        last_name = last.get("name")
        if last_name == "ask_clarify" and float(getattr(state, "conf", 0.0)) >= 0.6:
            cands = [a for a in cands if a.name != "ask_clarify"] + \
                    [Action(kind="policy", name="ask_clarify", args={"hint": "short"})]

    return _dedup(cands)

if __name__ == "__main__":
    s = State(s=[0.1]*8, u=0.2, conf=0.8)
    print("greeting:", [a.name for a in generate(s, {"intent": "greeting"})])
    print("compute(args):", [a.args for a in generate(s, {"intent":"compute","args":{"expr":"12*(3+1)"}}) if a.name=="invoke_calc"])
    print("compute(raw):", [a.args for a in generate(s, {"intent":"compute","args":{"raw":"حاصل 7*(5-2)؟"}}) if a.name=="invoke_calc"])
    print("unknown:", [a.name for a in generate(s, {"intent":"unknown"})])
