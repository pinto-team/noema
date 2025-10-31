# -*- coding: utf-8 -*-
"""
NOEMA • lang/parse.py — تحلیل نیت (Intent) و استخراج آرگومان‌ها از متن (V0 سبک)

هدف:
  - متن ورودی کاربر را به یک «طرح/نیت» ساده تبدیل کند تا به کنترل‌گر داده شود.
  - نیت‌های پشتیبانی‌شده در V0:  [greeting, compute, unknown]
  - برای compute، تلاش می‌کند عبارت عددی مجاز را استخراج کند.

خروجیِ اصلی:
    plan = parse(text: str, wm: Optional[WorkingMemory] = None) -> Dict[str, Any]
    # نمونه:
    # {"intent": "greeting", "args": {}, "confidence": 0.92}
    # {"intent": "compute",  "args": {"expr": "12*(3+1)" , "raw": "<متن>"}, "confidence": 0.85}
    # {"intent": "unknown",  "args": {"raw": "<متن>"}, "confidence": 0.40}

یادداشت‌ها:
  - این ماژول هیچ وابستگی سنگینی ندارد و از قوانین ساده/RegEx استفاده می‌کند.
  - نرمال‌سازی فارسی/عربی با perception.normalize_text (در صورت موجود بودن) انجام می‌شود.
  - اعتبارسنجی نهایی آرگومان‌ها در toolhub.verify صورت می‌گیرد.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import re

# نرمال‌سازی: اگر perception موجود بود از آن استفاده می‌کنیم
try:
    from perception import normalize_text  # type: ignore
except Exception:
    import unicodedata
    def normalize_text(t: str) -> str:
        if not t:
            return ""
        t = unicodedata.normalize("NFC", t)
        t = t.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")  # ي/ك → ی/ک
        t = t.replace("\u0640", " ").replace("\u200c", " ")            # کشیده/ZWNJ
        t = re.sub(r"\s+", " ", t).strip().lower()
        return t

# الگوهای ساده‌ی سلام
_GREET_WORDS = [
    "سلام", "درود", "سلاممم", "سلااام",  # فارسی
    "hi", "hello", "hey", "yo", "hola",   # چند زبان متداول
]
_GREET_RE = re.compile("|".join([re.escape(w) for w in _GREET_WORDS]), re.IGNORECASE)

# الگوهای اشاره به محاسبه
_COMPUTE_HINT_RE = re.compile(r"(حاصل|محاسبه|برابر|چند می(?:شود|شه)|جواب|نتیجه)", re.IGNORECASE)

# عبارت عددی مجاز (هم‌راستا با toolhub.verify و candidates)
_EXPR_RE = re.compile(r"[0-9+\-*/() \t]+")

def _has_any_math_symbol(text: str) -> bool:
    return any(ch in text for ch in "+-*/()")

def _extract_expr(text: str) -> Optional[str]:
    m = re.search(r"([0-9+\-*/() \t]{2,})", text)
    if not m:
        return None
    expr = (m.group(1) or "").strip()
    if not expr:
        return None
    # فقط کاراکترهای مجاز
    if not _EXPR_RE.fullmatch(expr):
        return None
    return expr

# ─────────────────────────── تشخیص نیت ───────────────────────────

def detect_intent(text: str) -> Dict[str, Any]:
    """
    تحلیل اولیه‌ی نیت و جمع‌آوری شواهد.
    خروجی: {"intent": "...", "confidence": 0.xx, "args": {...}}
    """
    raw = text or ""
    t = normalize_text(raw)

    # 1) سلام/احوالپرسی
    if _GREET_RE.search(t):
        return {"intent": "greeting", "confidence": 0.92, "args": {}}

    # 2) محاسبه‌ی عددی
    #    قواعد: وجود نشانه‌های ریاضی یا کلمات راهنما یا وجود یک عبارت قابل‌استخراج
    expr = _extract_expr(t)
    if expr or _has_any_math_symbol(t) or _COMPUTE_HINT_RE.search(t):
        if not expr:
            # ممکن است کاربر فقط بگوید "حاصل جمع ۷ و ۵" ولی ارقام لاتین نباشند
            # در V0 از مربی می‌خواهیم بعداً روشن کند؛ raw را نگه می‌داریم
            return {"intent": "compute", "confidence": 0.70, "args": {"raw": raw}}
        return {"intent": "compute", "confidence": 0.85, "args": {"expr": expr, "raw": raw}}

    # 3) ناشناخته
    return {"intent": "unknown", "confidence": 0.40, "args": {"raw": raw}}

# ─────────────────────────── API اصلی ───────────────────────────

def parse(text: str, wm: Optional[Any] = None) -> Dict[str, Any]:
    """
    نقطه‌ی ورود عمومی. فعلاً wm استفاده نمی‌شود اما برای آینده نگه داشته شده
    (مثلاً: استفاده از زمینه‌ی اخیر برای تخمین بهتر نیت).
    """
    plan = detect_intent(text)
    return plan

# ─────────────────────────── اجرای مستقیم (تست سریع) ───────────────────────────

if __name__ == "__main__":
    samples = [
        "سلام", "درود دوست من", "hi there!",
        "۲+۲؟", "حاصل 7*(5-2) چند می‌شود؟", "یه حساب انجام بده",
        "می‌خوام بدونم آب و هوا چطوره",
    ]
    for s in samples:
        p = parse(s)
        print(f"{s!r} -> {p}")
