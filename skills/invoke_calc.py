# -*- coding: utf-8 -*-
"""NOEMA • skills/invoke_calc.py — Safe calculator skill (improved V0)."""

from __future__ import annotations

from typing import Any, Dict, Optional
import re
import unicodedata

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

# ------------------------ Regex / Maps ------------------------
_ASCII_EXPR_RE = re.compile(r"^[0-9+\-*/() \t]+$")              # final safety gate
_NUM_CHUNK_RE = re.compile(r"([0-9+\-*/() \t]{2,})")            # to extract first expr chunk

# Persian/Arabic digits -> ASCII
_ARABIC_INDIC  = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EXT_ARABIC_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Zero-width / bidi chars
_ZW_RE = re.compile(r"[\u200c\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069]")

# ------------------------ Normalizers ------------------------
def _strip_zw(s: str) -> str:
    return _ZW_RE.sub("", s)

def _to_ascii_math(expr: str) -> str:
    """Convert localized digits/symbols + common FA math words → ASCII operators."""
    if not isinstance(expr, str):
        return ""

    # base normalization
    out = expr
    out = unicodedata.normalize("NFC", out)
    out = _strip_zw(out)
    out = out.translate(_EXT_ARABIC_INDIC).translate(_ARABIC_INDIC)

    # symbol-level
    out = (out.replace("×", "*")
              .replace("÷", "/")
              .replace("−", "-")
              .replace("–", "-")
              .replace("—", "-"))

    # phrase-level (order matters; use word-boundary / context-aware replacements)
    # ضرب/ضربدر → *
    out = re.sub(r"\bضربدر\b", "*", out)
    out = re.sub(r"\bضرب\b", "*", out)

    # تقسیم بر / تقسیم → /
    out = re.sub(r"\bتقسیم\s*بر\b", "/", out)
    out = re.sub(r"\bتقسیم\b", "/", out)

    # 'بر' فقط وقتی بین دو عدد است → /
    out = re.sub(r"(?<=\d)\s*بر\s*(?=\d)", "/", out)

    # به علاوه / جمع (بین اعداد) → +
    out = re.sub(r"\bبه\s*علاوه\b", "+", out)
    out = re.sub(r"(?<=\d)\s*جمع\s*(?=\d)", "+", out)

    # منهای → -
    out = re.sub(r"\bمنهای\b", "-", out)

    # منفی n → -n
    out = re.sub(r"\bمنفی\s*(\d+)", r"-\1", out)

    # squeeze spaces
    out = re.sub(r"\s+", " ", out).strip()
    return out

def _first_expr(s: str) -> Optional[str]:
    """Extract the first valid ASCII math expression after normalization."""
    if not isinstance(s, str):
        return None
    norm = _to_ascii_math(s)
    m = _NUM_CHUNK_RE.search(norm)
    if not m:
        return None
    expr = (m.group(1) or "").strip()
    if not expr:
        return None
    if not _ASCII_EXPR_RE.fullmatch(expr):
        return None
    return expr

# ------------------------ Eval ------------------------
def _safe_eval(expr: str) -> str:
    """Extremely limited eval: ASCII digits and + - * / ( ) and whitespace only."""
    if not isinstance(expr, str):
        raise ValueError("expr must be str")
    expr = _to_ascii_math(expr).strip()
    if not _ASCII_EXPR_RE.fullmatch(expr):
        raise ValueError("invalid characters in expression")

    # eval in empty builtins; our regex makes this safe.
    try:
        result = eval(expr, {"__builtins__": {}}, {})
    except ZeroDivisionError as e:
        raise ZeroDivisionError("division by zero") from e

    # pretty result: drop trailing .0
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)

_DEFAULT_META_OK  = {"confidence": 0.90, "u": 0.10, "r_total": 0.0, "risk": 0.0}
_DEFAULT_META_ERR = {"confidence": 0.65, "u": 0.35, "r_total": -0.1, "risk": 0.0}

# ------------------------ Extraction from plan/user text ------------------------
def _extract_expr_from_plan_or_text(plan: Optional[Dict[str, Any]], user_text: str) -> Optional[str]:
    # 1) From plan.args
    if isinstance(plan, dict):
        args = plan.get("args") or {}
        expr = args.get("expr")
        if isinstance(expr, str) and expr.strip():
            expr_norm = _to_ascii_math(expr.strip())
            if _ASCII_EXPR_RE.fullmatch(expr_norm):
                return expr_norm
        raw = args.get("raw")
        if isinstance(raw, str):
            cand = _first_expr(raw)
            if cand:
                return cand

    # 2) From user_text (FA/EN free-form)
    if isinstance(user_text, str):
        cand = _first_expr(user_text)
        if cand:
            return cand

    return None

# ------------------------ Public API ------------------------
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
            "extras": {"expr_ascii": "", **dict(extras or {})},
            "label_ok": False,
        }

    result: Optional[str] = None
    registry_source = "none"

    if tool_registry is None and load_registry is not None:
        try:
            tool_registry = load_registry()
        except Exception:
            tool_registry = None

    if tool_registry is not None:
        try:
            result = str(tool_registry.invoke("invoke_calc", expr=expr))
            registry_source = "toolhub"
        except Exception:
            result = None  # fall back

    if result is None:
        try:
            result = _safe_eval(expr)
            registry_source = "local"
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
                "extras": {"expr_ascii": expr, "source": registry_source, **dict(extras or {})},
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
        "extras": {"expr_ascii": expr, "source": registry_source, **dict(extras or {})},
        "label_ok": True,
    }

# ------------------------ Quick self-test ------------------------
if __name__ == "__main__":
    print(run(user_text="answer 7*(5-2)")["text_out"])                 # EN
    print(run(plan={"intent":"compute","args":{"expr":"2+2"}})["text_out"])
    print(run(user_text="۹ تقسیم بر ۳")["text_out"])                   # → 9 / 3 = 3
    print(run(user_text="3 ضرب 4")["text_out"])                        # → 3 * 4 = 12
    print(run(user_text="2+2=")["text_out"])                           # → 2+2 = 4
    print(run(user_text="حاصل ۲ به علاوه ۳؟")["text_out"])             # → 2+3 = 5
