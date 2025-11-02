# -*- coding: utf-8 -*-
"""NOEMA • skills/ask_clarify.py — Clarification skill (V0, strict-safe)"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

try:
    from lang.format import load_style, format_reply, Style  # type: ignore
except Exception:
    class Style:
        def __init__(self) -> None:
            self.formal = False
            self.max_len = 500
            self.prefix_emoji = False
            self.show_confidence = False
            self.tone = "friendly"
    def load_style(*args, **kwargs) -> Style:  # type: ignore
        return Style()
    def format_reply(*, intent: str, outcome: Dict[str, Any] | None = None,
                     style: Optional[Style] = None, meta: Optional[Dict[str, Any]] = None) -> str:  # type: ignore
        return (outcome or {}).get("message", "Please clarify your goal.")

_DEFAULT_META = {"confidence": 0.78, "u": 0.35, "risk": 0.0, "r_total": 0.0}

def _norm(plan: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    args = dict(((plan or {}).get("args") or {}))
    args.update({k: v for k, v in kwargs.items() if v is not None})
    hint = str(args.get("hint", "") or "").strip().lower()
    if hint not in ("short", "detail", ""): hint = ""
    reason = str(args.get("reason", "") or "").strip()
    ex = args.get("examples", [])
    if ex is None: examples: List[str] = []
    elif isinstance(ex, (list, tuple)): examples = [str(x) for x in ex if str(x).strip()][:3]
    else: examples = [str(ex)]
    return {"hint": hint, "reason": reason, "examples": examples}

def _compose(style: Style, hint: str, reason: str, examples: List[str]) -> str:
    base = "Please clarify your goal so I can help precisely." if style.formal else "Could you clarify what you want me to do?"
    lines = [base]
    if hint == "short":
        lines.append("One short sentence is enough.")
    else:
        lines += ["Please specify:",
                  "• Goal (expected outcome)",
                  "• Input constraints",
                  "• Output format (text/JSON/markdown)"]
    if examples: lines.append(f'For example: "{examples[0]}"')
    if reason: lines.append(f"(reason: {reason})")
    return format_reply(intent="clarify", outcome={"message": "\n".join(lines)}, style=style, meta=_DEFAULT_META)

def run(user_text: str = "", *, plan: Optional[Dict[str, Any]] = None,
        style: Optional[Style] = None, extras: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    style = style or load_style()
    a = _norm(plan, kwargs)
    text_out = _compose(style, a["hint"], a["reason"], a["examples"])
    meta = dict(_DEFAULT_META)
    if a["hint"] or a["examples"]:
        meta["confidence"] = min(0.85, meta["confidence"] + 0.05)
        meta["u"] = max(0.25, meta["u"] - 0.05)
    return {
        "intent": "clarify",
        "outcome": a,
        "text_out": text_out,
        "meta": meta,
        "tests": [{"name": "clarify_prompted", "pass": True}],
        "extras": dict(extras or {}),
        "label_ok": True,
    }

if __name__ == "__main__":
    print(run(user_text="?", plan={"args": {"hint": "short"}})["text_out"])
