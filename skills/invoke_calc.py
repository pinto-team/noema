# -*- coding: utf-8 -*-
"""NOEMA • skills/invoke_calc.py — Safe calculator skill (V0).

- Evaluates a simple arithmetic expression in a very restricted sandbox.
- If a tool registry is available, it prefers invoking `invoke_calc` there;
  otherwise it falls back to local safe eval.

Contract:
    run(
        user_text: str = "",
        *,
        plan: dict | None = None,      # {"intent":"compute","args":{"expr": "..."}}
        style: "lang.format.Style" | None = None,
        tool_registry: "toolhub.ToolRegistry" | None = None,
        extras: dict | None = None,
        **kwargs
    ) -> dict
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import re

# --- Style/formatter (optional) ---
try:
    from lang.format import load_style, format_reply, Style  # type: ignore
except Exception:
    class Style:  # minimal fallback
        def __init__(self) -> None:
            self.formal = False
            self.max_len = 500
            self.prefix_emoji = False
            self.show_confidence = False
            self.tone = "friendly"
    def load_style(*args, **kwargs) -> Style:  # type: ignore
        return Style()
    def format_reply(*, intent: str, outcome: Dict[str, Any] | None = None, style: Optional[Style] = None, meta: Optional[Dict[str, Any]] = None) -> str:  # type: ignore
        style = style or Style()
        oc = outcome or {}
        expr, res = oc.get("expr", ""), oc.get("result", "")
        return f"{expr} = {res}" if expr else f"Result: {res}"

# --- Optional tool registry ---
try:
    from toolhub import load_registry  # type: ignore
except Exception:
    load_registry = None  # type: ignore

_ASCII_EXPR_RE = re.compile(r"^[0-9+\-*/() \t]+$")

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EXT_ARABIC_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

def _to_ascii_math(expr: str) -> str:
    if not isinstance(expr, str):
        return ""
    out = expr.translate(_EXT_ARABIC_INDIC).translate(_ARABIC_INDIC)
    out = (
        out.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    return out

def _safe_eval(expr: str) -> str:
    """Extremely limited eval: ASCII digits and + - * / ( ) and whitespace only."""
    if not isinstance(expr, str):
        raise ValueError("expr must be str")
    expr = _to_ascii_math(expr).strip()
    if not _ASCII_EXPR_RE.fullmatch(expr):
        raise ValueError("invalid characters in expression")
    return str(eval(expr, {"__builtins__": {}}, {}))

_DEFAULT_META_OK = {"confidence": 0.88, "u": 0.12, "r_total": 0.0, "risk": 0.0}
_DEFAULT_META_ERR = {"confidence": 0.65, "u": 0.35, "r_total": -0.1, "risk": 0.0}

def _extract_expr_from_plan_or_text(plan: Optional[Dict[str, Any]], user_text: str) -> Optional[str]:
    # 1) From plan.args
    if isinstance(plan, dict):
        args = plan.get("args") or {}
        expr = args.get("expr")
        if isinstance(expr, str) and expr.strip():
            return expr.strip()
        raw = args.get("raw")
        if isinstance(raw, str):
            m = re.search(r"([0-9+\-*/() \t]{2,})", _to_ascii_math(raw))
            if m:
                cand = m.group(1).strip()
                if _ASCII_EXPR_RE.fullmatch(cand):
                    return cand
    # 2) From user_text
    if isinstance(user_text, str):
        m = re.search(r"([0-9+\-*/() \t]{2,})", _to_ascii_math(user_text))
        if m:
            cand = m.group(1).strip()
            if _ASCII_EXPR_RE.fullmatch(cand):
                return cand
    return None

def run(
    user_text: str = "",
    *,
    plan: Optional[Dict[str, Any]] = None,
    style: Optional[Style] = None,
    tool_registry: Optional[Any] = None,
    extras: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run the safe calculator skill."""
    style = style or load_style()
    expr = _extract_expr_from_plan_or_text(plan, user_text)

    if not expr:
        text_out = "Please provide an arithmetic expression like '7*(5-2)'."
        return {
            "intent": "compute",
            "outcome": {"expr": "", "result": ""},
            "text_out": text_out,
            "meta": dict(_DEFAULT_META_ERR),
            "extras": dict(extras or {}),
            "label_ok": False,
        }

    result: Optional[str] = None
    if tool_registry is None and load_registry is not None:
        try:
            tool_registry = load_registry()
        except Exception:
            tool_registry = None

    if tool_registry is not None:
        try:
            result = str(tool_registry.invoke("invoke_calc", expr=expr))
        except Exception:
            result = None  # fall back

    if result is None:
        try:
            result = _safe_eval(expr)
        except Exception as e:
            outcome = {"expr": expr, "result": "invalid expression"}
            meta = dict(_DEFAULT_META_ERR)
            meta["error"] = str(e)
            text_out = format_reply(intent="compute", outcome=outcome, style=style, meta=meta)
            return {
                "intent": "compute",
                "outcome": outcome,
                "text_out": text_out,
                "meta": meta,
                "extras": dict(extras or {}),
                "label_ok": False,
            }

    outcome = {"expr": expr, "result": str(result)}
    meta = dict(_DEFAULT_META_OK)
    text_out = format_reply(intent="compute", outcome=outcome, style=style, meta=meta)

    return {
        "intent": "compute",
        "outcome": outcome,
        "text_out": text_out,
        "meta": meta,
        "extras": dict(extras or {}),
        "label_ok": True,
    }

if __name__ == "__main__":
    print(run(user_text="answer 7*(5-2)")["text_out"])  # demo
    print(run(plan={"intent":"compute","args":{"expr":"2+2"}})["text_out"])