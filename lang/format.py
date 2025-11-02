# -*- coding: utf-8 -*-
"""
NOEMA • lang/format.py — Lightweight response formatting (V0)

Purpose:
  - Turn (intent, outcome) into the final user-facing text.
  - Default style: friendly/neutral English; configurable via config/meta.yaml.
  - No heavy deps; YAML optional.

API:
    from lang.format import load_style, format_reply

    style = load_style()
    txt = format_reply(
        intent="compute",
        outcome={"result": "4", "expr": "2+2"},
        style=style,
        meta={"confidence": 0.92},
    )

Contract:
  - intent:  "greeting" | "compute" | "clarify" | "unknown" | ...
  - outcome: dict produced by a skill/policy/tool
      * greeting: {"variant": "default"}
      * compute : {"expr": "<str>", "result": "<str>"}
      * clarify : {"hint": "short" | "detail"}
      * unknown : {"note": "..."} (optional)
  - If outcome["text_out"] exists, it's used as a fallback verbatim.

Notes:
  - Outputs are largely deterministic for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import re

# Optional YAML for style
try:
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


# --------- Normalization (language-agnostic) ---------
def _normalize(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ----------------------------- Style -----------------------------
@dataclass
class Style:
    tone: str = "friendly"  # "friendly" | "neutral"
    formal: bool = False
    max_len: int = 500
    prefix_emoji: bool = False
    show_confidence: bool = False


def load_style(path: str | Path = "config/meta.yaml") -> Style:
    p = Path(path)
    if not p.exists():
        return Style()
    try:
        if _HAS_YAML:
            obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        else:
            obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        obj = {}
    st = Style()
    if isinstance(obj, dict):
        st.tone = str(obj.get("tone", st.tone))
        st.formal = bool(obj.get("formal", st.formal))
        st.max_len = int(obj.get("max_len", st.max_len))
        st.prefix_emoji = bool(obj.get("prefix_emoji", st.prefix_emoji))
        st.show_confidence = bool(obj.get("show_confidence", st.show_confidence))
    return st


# ----------------------------- Renderers -----------------------------
def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else (s[: max(0, n - 1)].rstrip() + "…")


def _decorate_conf(text: str, meta: Dict[str, Any], style: Style) -> str:
    if style.show_confidence:
        conf = meta.get("confidence")
        if isinstance(conf, (int, float)):
            return f"{text}\n\n[Confidence: {conf:.2f}]"
    return text


def _render_greeting(style: Style, outcome: Dict[str, Any]) -> str:
    base = "Hello." if style.formal else "Hello! Welcome."
    if style.prefix_emoji and not style.formal:
        base = "👋 " + base
    return base


def _render_compute(style: Style, outcome: Dict[str, Any]) -> str:
    expr = _normalize(str(outcome.get("expr", "")).strip())
    res = str(outcome.get("result", "")).strip()
    if not expr and "text_out" in outcome:
        return str(outcome["text_out"])

    if style.formal:
        if expr:
            return f"The result of {expr} is {res}"
        return f"Result: {res}"
    else:
        if expr:
            return f"{expr} = {res}"
        return f"The answer is {res}"


def _render_clarify(style: Style, outcome: Dict[str, Any]) -> str:
    hint = str(outcome.get("hint", "") or "")
    if style.formal:
        return "Could you please clarify your request?" if hint == "short" else \
               "To help better, please clarify what you would like me to do."
    else:
        return "Could you clarify what you mean?" if hint == "short" else \
               "Please share a bit more detail so I can help better."


def _render_unknown(style: Style, outcome: Dict[str, Any]) -> str:
    if style.formal:
        return "I did not fully understand. Would you like me to compute something or answer a different question?"
    else:
        return "I didn't quite get that. Should I calculate something or help with a different request?"


# ----------------------------- Public API -----------------------------
def format_reply(
    *,
    intent: str,
    outcome: Optional[Dict[str, Any]] = None,
    style: Optional[Style] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    intent + outcome -> final text.
    If outcome['text_out'] exists, it takes precedence as a fallback.
    """
    style = style or Style()
    meta = dict(meta or {})
    outcome = dict(outcome or {})

    ready = outcome.get("text_out")
    if isinstance(ready, str) and ready.strip():
        return _truncate(_decorate_conf(ready.strip(), meta, style), style.max_len)

    it = (intent or "unknown").strip().lower()

    if it == "greeting":
        txt = _render_greeting(style, outcome)
    elif it == "compute":
        txt = _render_compute(style, outcome)
    elif it == "clarify":
        txt = _render_clarify(style, outcome)
    else:
        txt = _render_unknown(style, outcome)

    txt = _decorate_conf(txt, meta, style)
    return _truncate(txt, style.max_len)


# ----------------------------- Quick self-test -----------------------------
if __name__ == "__main__":
    st = load_style()
    print(format_reply(intent="greeting", outcome={}, style=st, meta={"confidence": 0.93}))
    print(format_reply(intent="compute", outcome={"expr": "2+2", "result": "4"}, style=st))
    print(format_reply(intent="clarify", outcome={"hint": "short"}, style=Style(formal=False)))
    print(format_reply(intent="unknown", outcome={}, style=Style(formal=True, show_confidence=True), meta={"confidence": 0.41}))
