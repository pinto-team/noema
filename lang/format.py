# -*- coding: utf-8 -*-
"""
NOEMA • lang/format.py — قالب‌بندی خروجی متنی نوما (V0 سبک و قابل‌تنظیم)

هدف:
  - یک لایه‌ی کوچک برای تبدیل «نتیجه‌ی کنش/نیت» به متن نهایی برای کاربر.
  - زبان پیش‌فرض: فارسی (formal=ملایم/مودب)، با گزینه‌ی سبک دوستانه.
  - بدون وابستگی سنگین؛ YAML اختیاری برای تنظیمات (config/meta.yaml).

API اصلی:
    from lang.format import load_style, format_reply

    style = load_style()  # از config/meta.yaml اگر موجود باشد
    txt = format_reply(
        intent="compute",
        outcome={"result": "4", "expr": "2+2"},
        style=style,
        meta={"confidence": 0.92}
    )

پیمان داده:
  - intent:  "greeting" | "compute" | "clarify" | "unknown"
  - outcome: دیکشنری خروجی کنش/سیاست؛ کلیدهای متداول:
      * greeting: {"variant": "default"}
      * compute : {"expr": "<str>", "result": "<str>"}
      * clarify : {"hint": "short" | "detail"}
      * unknown : {"note": "..."} (اختیاری)
  - اگر outcome["text_out"] وجود داشته باشد، به‌عنوان fallback استفاده می‌شود.

یادداشت:
  - برای تست‌های خودکار، خروجی‌ها عمدتاً قطعی هستند (بدون تصادفی).
  - اگر perception.normalize_text در دسترس باشد، ابزار normalize درونی از آن
    استفاده می‌کند تا فاصله/کاراکترهای عربی/فارسی یک‌دست شوند.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from pathlib import Path
import json
import re

# --- نرمال‌سازی سبک ---
try:
    from perception import normalize_text as _normalize  # type: ignore
except Exception:
    def _normalize(t: str) -> str:
        if not t: return ""
        t = t.replace("\u064a","\u06cc").replace("\u0643","\u06a9")  # ي/ك→ی/ک
        t = t.replace("\u0640", " ").replace("\u200c"," ")           # کشیده/ZWNJ
        t = re.sub(r"\s+", " ", t).strip()
        return t

# --- YAML اختیاری برای استایل ---
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# ----------------------------- Style -----------------------------

@dataclass
class Style:
    tone: str = "friendly"         # "friendly" | "neutral"
    formal: bool = False           # اگر True → لحن رسمی‌تر
    max_len: int = 500             # حداکثر طول متن خروجی
    prefix_emoji: bool = False     # اگر True → در سلام یک ایموجی کوچک می‌افزاید
    show_confidence: bool = False  # برای دیباگ: نمایش (اختیاری) اعتمادبه‌نفس

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

# ----------------------------- رندرهای نیت‌ها -----------------------------

def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else (s[:max(0, n-1)].rstrip() + "…")

def _decorate_conf(text: str, meta: Dict[str, Any], style: Style) -> str:
    if style.show_confidence:
        conf = meta.get("confidence")
        if isinstance(conf, (int, float)):
            return f"{text}\n\n[اطمینان: {conf:.2f}]"
    return text

def _render_greeting(style: Style, outcome: Dict[str, Any]) -> str:
    if style.formal:
        base = "درود بر شما."
    else:
        base = "سلام! خوش اومدی."
    if style.prefix_emoji and not style.formal:
        base = "👋 " + base
    return base

def _render_compute(style: Style, outcome: Dict[str, Any]) -> str:
    expr = _normalize(str(outcome.get("expr", "")).strip())
    res  = str(outcome.get("result", "")).strip()
    if not expr and "text_out" in outcome:
        return str(outcome["text_out"])
    # لحن
    if style.formal:
        if expr:
            return f"نتیجهٔ {expr} = {res}"
        return f"نتیجه: {res}"
    else:
        if expr:
            return f"{expr} = {res}"
        return f"جوابش می‌شود: {res}"

def _render_clarify(style: Style, outcome: Dict[str, Any]) -> str:
    hint = str(outcome.get("hint", "") or "")
    if style.formal:
        if hint == "short":
            return "منظورتان را کمی دقیق‌تر بیان می‌کنید؟"
        return "برای کمک بهتر، لطفاً منظورتان را دقیق‌تر توضیح دهید."
    else:
        if hint == "short":
            return "دقیق‌تر می‌گی چی می‌خوای؟"
        return "برای اینکه بهتر کمک کنم، لطفاً واضح‌تر بگو چی مدنظرته."

def _render_unknown(style: Style, outcome: Dict[str, Any]) -> str:
    if style.formal:
        return "دقیق متوجه نشدم. می‌خواهید محاسبه انجام دهم یا پرسش دیگری دارید؟"
    else:
        return "هنوز کامل متوجه نشدم. حساب انجام بدم یا چیز دیگه‌ای مدنظرته؟"

# ----------------------------- API اصلی -----------------------------

def format_reply(
    *,
    intent: str,
    outcome: Optional[Dict[str, Any]] = None,
    style: Optional[Style] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    intent + outcome → متن نهایی.
    اگر outcome["text_out"] وجود داشت، در اولویت به‌عنوان fallback استفاده می‌شود.
    """
    style = style or Style()
    meta = dict(meta or {})
    outcome = dict(outcome or {})

    # اگر متن آماده داده شده بود
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

# ----------------------------- اجرای مستقیم (تست سریع) -----------------------------

if __name__ == "__main__":
    st = load_style()
    print(format_reply(intent="greeting", outcome={}, style=st, meta={"confidence":0.93}))
    print(format_reply(intent="compute", outcome={"expr":"2+2","result":"4"}, style=st))
    print(format_reply(intent="clarify", outcome={"hint":"short"}, style=Style(formal=False)))
    print(format_reply(intent="unknown", outcome={}, style=Style(formal=True, show_confidence=True), meta={"confidence":0.41}))
