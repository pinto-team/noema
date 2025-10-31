# -*- coding: utf-8 -*-
"""
NOEMA • skills/invoke_calc.py — مهارت «ماشین‌حساب امن» (V0)

هدف:
  - یک عبارت عددی ساده (۰-۹، + - * / پرانتز) را «ایمن» ارزیابی می‌کند.
  - از رجیستری toolhub (اگر موجود باشد) برای اجرای ابزار `invoke_calc` استفاده می‌کند؛
    وگرنه از ارزیاب داخلیِ محدود بهره می‌برد.

قرارداد run():
    run(
        user_text: str = "",
        *,
        plan: dict | None = None,      # {"intent":"compute","args":{"expr": "..."}}
        style: "lang.format.Style" | None = None,
        tool_registry: "toolhub.ToolRegistry" | None = None,
        extras: dict | None = None,
        **kwargs
    ) -> dict

خروجی dict:
    {
      "intent": "compute",
      "outcome": {"expr": "<str>", "result": "<str>"},
      "text_out": "<متن آماده نمایش>",
      "meta": {"confidence": 0.88, "u": 0.12, "r_total": 0.0, "risk": 0.0},
      "extras": {...},
      "label_ok": True/False   # در صورت خطا → False
    }

نکته‌های ایمنی:
  - فقط کاراکترهای مجاز: ارقام، چهار عمل اصلی، پرانتز و فاصله/تب.
  - هیچ نام/تابع/ماژول پایتونی اجازه ندارد (eval با محیط خالی).
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import re

# --- زبان/قالب‌بندی ---
try:
    from lang import load_style, format_reply, Style  # type: ignore
except Exception:
    class Style:
        def __init__(self): self.formal=False; self.max_len=500; self.prefix_emoji=False; self.show_confidence=False; self.tone="friendly"
    def load_style(*args, **kwargs): return Style()
    def format_reply(*, intent: str, outcome: Dict[str, Any] | None = None, style: Optional[Style] = None, meta: Optional[Dict[str, Any]] = None) -> str:
        style = style or Style()
        oc = outcome or {}
        expr, res = oc.get("expr",""), oc.get("result","")
        return f"{expr} = {res}" if expr else f"نتیجه: {res}"

# --- رجیستری ابزار (اختیاری) ---
try:
    from toolhub import load_registry  # type: ignore
except Exception:
    load_registry = None  # fallback به ارزیاب داخلی

_EXPR_RE = re.compile(r"^[0-9+\-*/() \t]+$")

def _safe_eval(expr: str) -> str:
    """
    ارزیابی بسیار محدود: فقط 0-9، + - * / ( ) و فاصله.
    """
    if not isinstance(expr, str):
        raise ValueError("expr must be str")
    expr = expr.strip()
    if not _EXPR_RE.fullmatch(expr):
        raise ValueError("invalid characters in expression")
    # eval با محیط خالی
    return str(eval(expr, {"__builtins__": {}}, {}))

_DEFAULT_META_OK = {"confidence": 0.88, "u": 0.12, "r_total": 0.0, "risk": 0.0}
_DEFAULT_META_ERR = {"confidence": 0.65, "u": 0.35, "r_total": -0.1, "risk": 0.0}

def _extract_expr_from_plan_or_text(plan: Optional[Dict[str, Any]], user_text: str) -> Optional[str]:
    # 1) از plan.args
    if isinstance(plan, dict):
        args = plan.get("args") or {}
        expr = args.get("expr")
        if isinstance(expr, str) and expr.strip():
            return expr.strip()
        raw = args.get("raw")
        if isinstance(raw, str):
            m = re.search(r"([0-9+\-*/() \t]{2,})", raw)
            if m:
                cand = m.group(1).strip()
                if _EXPR_RE.fullmatch(cand):
                    return cand
    # 2) از متن کاربر
    if isinstance(user_text, str):
        m = re.search(r"([0-9+\-*/() \t]{2,})", user_text)
        if m:
            cand = m.group(1).strip()
            if _EXPR_RE.fullmatch(cand):
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
    """
    اجرای مهارت ماشین‌حساب امن.
    """
    style = style or load_style()
    expr = _extract_expr_from_plan_or_text(plan, user_text)

    # اگر عبارت پیدا نشد → پیام شفاف‌سازی
    if not expr:
        text_out = "برای محاسبه، یک عبارت عددی مثل «7*(5-2)» بده."
        return {
            "intent": "compute",
            "outcome": {"expr": "", "result": ""},
            "text_out": text_out,
            "meta": dict(_DEFAULT_META_ERR),
            "extras": dict(extras or {}),
            "label_ok": False,
        }

    # تلاش با رجیستری ابزار (اگر در دسترس)
    result: Optional[str] = None
    if tool_registry is None and load_registry is not None:
        try:
            tool_registry = load_registry()  # invoke_calc پیش‌فرض بایند می‌شود
        except Exception:
            tool_registry = None

    if tool_registry is not None:
        try:
            result = str(tool_registry.invoke("invoke_calc", expr=expr))
        except Exception:
            result = None  # به fallback داخلی می‌رویم

    # fallback داخلی
    if result is None:
        try:
            result = _safe_eval(expr)
        except Exception as e:
            # خطا در ارزیابی
            outcome = {"expr": expr, "result": "خطای عبارت"}
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

    # موفق
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

# --- تست مستقیم ---
if __name__ == "__main__":
    print(run(user_text="جواب 7*(5-2) رو بگو")["text_out"])
    print(run(plan={"intent":"compute","args":{"expr":"2+2"}})["text_out"])
    print(run(plan={"intent":"compute","args":{"expr":"__import__('os').system('ls')"}})["text_out"])
