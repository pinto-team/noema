# -*- coding: utf-8 -*-
"""
NOEMA • control/candidates.py — تولید نامزدهای عمل (V0 سبک و قانون‌محور)

هدف:
  - بر اساس «نیت/plan» + نشانه‌های ساده از وضعیت، لیست کوتاهی از Actions بسازد.
  - این ماژول «تصمیم‌گیر» نیست؛ فقط چه گزینه‌هایی محتمل‌ترند را می‌چیند.
  - با app/main.py، control/policy.py و control/planner.py سازگار است.

ورودیِ generate:
  - state : world.State (برای دسترسی به conf/u در صورت نیاز)
  - plan  : dict شبیه {"intent":"compute","args":{"expr":"2+2"}}
  - wm    : (اختیاری) memory.WorkingMemory برای استفاده از زمینه‌ی اخیر
  - tool_registry : (اختیاری) رجیستری ابزارها (اگر داشتید)، تا ابزارهای مرتبط پیشنهاد شود

خروجی:
  - List[Action]  (بدون تکرار و با آرگومان‌های پر شده تا حد امکان)

یادداشت‌ها:
  - اگر intent نامشخص باشد → ask_clarify همیشه در فهرست است.
  - برای intent=compute اگر expr نداریم، از متنِ اخیر تلاشِ استخراج می‌کنیم.
  - اگر greeting تشخیص داده شود → reply_greeting در صدر است.
  - اگر tool_registry داشتید و intent ناشناخته بود، چند ابزار عمومی «ایمن» هم پیشنهاد می‌شود.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

# --- انواع داده (سازگار با world.*) ---
try:
    from world import State, Action  # type: ignore
except Exception:
    @dataclass
    class State:
        s: List[float]; u: float = 0.0; conf: float = 0.0
    @dataclass
    class Action:
        kind: str; name: str; args: Dict[str, Any]

# --- کمک‌های محلی ---
_SAFE_DEFAULT_TOOLS = [
    # اگر رجیستری ابزار دارید و ناشناخته بود، این‌ها را می‌توانید پیشنهاد دهید
    # این‌ها صرفاً نمونه‌نام هستند؛ با نام ابزارهای خودتان جایگزین کنید.
    ("tool", "invoke_calc", {}),           # ماشین‌حساب امن
    ("policy", "ask_clarify", {}),         # پرسش روشن‌ساز
]

_NUM_EXPR_RE = re.compile(r"([0-9+\-*/() \t]+)")

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
    m = _NUM_EXPR_RE.search(text or "")
    if m:
        expr = m.group(1).strip()
        # محافظت ساده: فقط کاراکترهای مجاز
        if re.fullmatch(r"[0-9+\-*/() \t]+", expr):
            return expr
    return None

def _context_tail_text(wm, k: int = 3) -> str:
    if wm is None:
        return ""
    pairs = wm.context(k=k)
    # آخرین ورودی کاربر (اگر بود)
    return pairs[-1][0] if pairs else ""

# --- API اصلی ---

def generate(
    state: State,
    plan: Dict[str, Any],
    wm: Optional[Any] = None,
    tool_registry: Optional[Any] = None,
) -> List[Action]:
    """
    تولید نامزدها بر اساس intent و زمینه.
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
            # تلاش استخراج از plan.raw یا متن اخیر
            raw = args.get("raw", "") or _context_tail_text(wm, k=3)
            expr = _maybe_extract_expr(raw) or "2+2"  # fallback بی‌ضرر
        cands.append(Action(kind="tool", name="invoke_calc", args={"expr": expr}))

        # گاهی clarify قبل از calc مفید است (اگر عدم‌قطعیت بالاست)
        if float(getattr(state, "u", 0.0)) >= 0.5:
            cands.append(Action(kind="policy", name="ask_clarify", args={}))

    # 3) intentهای دیگر می‌توانند در آینده اضافه شوند (search, summarize, ...)
    #    فعلاً همه‌ی موارد ناشناخته → clarify + گزینه‌های ایمن
    else:
        cands.append(Action(kind="policy", name="ask_clarify", args={}))
        # اگر رجیستری ابزار داریم، چند ابزار عمومی/ایمن پیشنهاد کنیم
        if tool_registry and hasattr(tool_registry, "list_safe_basics"):
            try:
                for tool_name in tool_registry.list_safe_basics():
                    cands.append(Action(kind="tool", name=tool_name, args={}))
            except Exception:
                pass
        else:
            for kind, name, a in _SAFE_DEFAULT_TOOLS:
                cands.append(Action(kind=kind, name=name, args=a))

    # 4) هیوریستیک‌های کوچک وابسته به حافظه‌ی کاری
    #    - اگر پاسخ قبلی هم clarify بوده و conf بالا است، تکرارِ clarify را کم کنیم
    if wm is not None and len(wm) > 0:
        last = wm.last_action() or {}
        last_name = last.get("name")
        if last_name == "ask_clarify" and float(getattr(state, "conf", 0.0)) >= 0.6:
            cands = [a for a in cands if a.name != "ask_clarify"] + \
                    [Action(kind="policy", name="ask_clarify", args={"hint":"short"})]  # نسخه‌ی کوتاه

    # 5) حذف تکراری‌ها و بازگرداندن
    return _dedup(cands)

# --- اجرای مستقیم برای تست دستی ---
if __name__ == "__main__":
    s = State(s=[0.1]*8, u=0.2, conf=0.8)

    # greeting
    print("greeting:", [a.name for a in generate(s, {"intent":"greeting"})])

    # compute با expr در args
    print("compute(args):", [a.args for a in generate(s, {"intent":"compute","args":{"expr":"12*(3+1)"}}) if a.name=="invoke_calc"])

    # compute بدون expr → استخراج از raw
    print("compute(raw):", [a.args for a in generate(s, {"intent":"compute","args":{"raw":"جواب 7*(5-2) رو بگو"}}) if a.name=="invoke_calc"])

    # unknown
    print("unknown:", [a.name for a in generate(s, {"intent":"unknown"})])
