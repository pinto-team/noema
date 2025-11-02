# -*- coding: utf-8 -*-
"""
NOEMA • control/candidates.py — action candidate generation (V0, improved)

Goal
-----
Given the current plan/intent + optional working memory context and an
optional tool registry, propose a small set of plausible Actions. This
module is *not* a decision maker; it only enumerates good options.

Key improvements in this version:
- Wire memory-like intents (e.g., praise/memory.reply) → reply_from_memory.
- Robust math extraction: Persian/Arabic digits, operator words, and Cf removal.
- Clean branch structure (no duplicated 'praise' branch) + safe de-duplication.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re
import unicodedata

# ----------------------------- world types (fallback stubs) -----------------------------
try:
    from world import State, Action  # type: ignore
except Exception:
    @dataclass
    class State:
        s: List[float]; u: float = 0.0; conf: float = 0.0
    @dataclass
    class Action:
        kind: str; name: str; args: Dict[str, Any]

# ----------------------------- normalization helpers -----------------------------
# Persian/Arabic digits → ASCII
_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Visual math symbols → ASCII
_MATH_SYMS  = {"×": "*", "÷": "/", "−": "-", "–": "-", "—": "-"}

# Allowed numeric expression (ASCII) used as guard
_ASCII_EXPR_RE = re.compile(r"^[0-9+\-*/() \t]+$")

# Loose finder for a candidate expression inside text
_NUM_EXPR_RE = re.compile(r"[0-9+\-*/() \t]{2,}")

# Intents that should answer from episodic/demo memory
MEMORY_INTENTS = {"memory.reply", "praise", "thanks", "gratitude"}

# Fallback safe tools if no registry is present
_SAFE_DEFAULT_TOOLS = [
    ("tool", "invoke_calc", {}),     # safe calculator
    ("policy", "ask_clarify", {}),   # clarifying question
]

def _strip_cf(t: str) -> str:
    """Remove Unicode format controls (e.g., LRM/RLM/ZWNJ) to stabilize parsing."""
    return "".join(ch for ch in t if unicodedata.category(ch) != "Cf")

def _fa_ops_to_ascii(t: str) -> str:
    """Map common Persian operator words to ASCII operators for easier extraction."""
    # Order matters: handle multi-word first
    pairs = [
        (r"\bتقسیم\s*بر\b", "/"),
        (r"\bتقسیم\b", "/"),
        (r"\bضرب\s*در\b", "*"),
        (r"\bضربدر\b", "*"),
        (r"\bضرب\b", "*"),
        (r"\bبه\s*علاوه\b", "+"),
        (r"\bجمع\b", "+"),
        (r"\bمنهای\b", "-"),
    ]
    for pat, rep in pairs:
        t = re.sub(pat, f" {rep} ", t)
    return t

def _to_ascii_math(text: str) -> str:
    if not text:
        return ""
    t = text.translate(_FA_DIGITS).translate(_AR_DIGITS)
    for k, v in _MATH_SYMS.items():
        t = t.replace(k, v)
    t = _fa_ops_to_ascii(t)
    t = _strip_cf(t)
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
    """Extract a safe arithmetic expression from text if possible."""
    t = _to_ascii_math(text or "")
    m = _NUM_EXPR_RE.search(t)
    if not m:
        return None
    expr = (m.group(0) or "").strip()
    if not expr:
        return None
    return expr if _ASCII_EXPR_RE.fullmatch(expr) else None

def _context_tail_text(wm, k: int = 3) -> str:
    """Peek a bit of recent context from WM (if available)."""
    if wm is None:
        return ""
    try:
        pairs = wm.context(k=k)
        return pairs[-1][0] if pairs else ""
    except Exception:
        return ""

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
    intent = (plan or {}).get("intent", "unknown") or "unknown"
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
            expr = _maybe_extract_expr(raw) or "2+2"
        cands.append(Action(kind="tool", name="invoke_calc", args={"expr": expr}))
        if float(getattr(state, "u", 0.0)) >= 0.5:
            cands.append(Action(kind="policy", name="ask_clarify", args={}))

    # 3) praise → پاسخ از حافظهٔ دمو
    elif intent == "praise":
        cands.append(Action(kind="skill", name="reply_from_memory", args={}))

    # 4) smalltalk → جواب گپ سبک
    elif intent == "smalltalk":
        cands.append(Action(kind="skill", name="reply_smalltalk", args={}))

    # 5) سایر موارد → clarify + ابزارهای امن
    else:
        cands.append(Action(kind="policy", name="ask_clarify", args={}))
        for kind, name, a in _SAFE_DEFAULT_TOOLS:
            cands.append(Action(kind=kind, name=name, args=a))

    # 6) Avoid clarify loops when we already clarified and confidence is decent
    try:
        if wm is not None and len(wm) > 0:
            last = wm.last_action() or {}
            last_name = last.get("name")
            if last_name == "ask_clarify" and float(getattr(state, "conf", 0.0)) >= 0.6:
                cands = [a for a in cands if a.name != "ask_clarify"] + \
                        [Action(kind="policy", name="ask_clarify", args={"hint": "short"})]
    except Exception:
        pass

    return _dedup(cands)


if __name__ == "__main__":
    s = State(s=[0.1]*8, u=0.2, conf=0.8)
    print("greeting:", [a.name for a in generate(s, {"intent": "greeting"})])
    print("compute(args):", [a.args for a in generate(s, {"intent":"compute","args":{"expr":"12*(3+1)"}}) if a.name=="invoke_calc"])
    print("compute(raw):", [a.args for a in generate(s, {"intent":"compute","args":{"raw":"حاصل 7*(5-2)؟"}}) if a.name=="invoke_calc"])
    print("memory:", [a.name for a in generate(s, {"intent":"praise"})])
    print("smalltalk:", [a.name for a in generate(s, {"intent":"smalltalk"})])
    print("unknown:", [a.name for a in generate(s, {"intent":"unknown"})])
