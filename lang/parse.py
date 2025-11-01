# -*- coding: utf-8 -*-
"""
NOEMA • lang/parse.py — تحلیل نیت (Intent) و استخراج آرگومان‌ها از متن (V0 سبک)

هدف:
  - متن ورودی کاربر را به یک «طرح/نیت» ساده تبدیل کند تا به کنترل‌گر داده شود.
  - نیت‌های پشتیبانی‌شده در V0:  [greeting, compute, smalltalk, memory.reply] + RULE/intent
  - برای compute، تلاش می‌کند عبارت عددی مجاز را استخراج کند.
  - اگر نمونهٔ DEMO مشابه وجود داشته باشد → memory.reply (برای مهارت حافظه)

خروجیِ اصلی:
    plan = parse(text: str, wm: Optional[WorkingMemory] = None) -> Dict[str, Any]
    # نمونه:
    # {"intent": "greeting", "args": {}, "confidence": 0.92}
    # {"intent": "compute",  "args": {"expr": "12*(3+1)" , "raw": "<متن>"}, "confidence": 0.85}
    # {"intent": "smalltalk",  "args": {"raw": "<متن>"}, "confidence": 0.60}

یادداشت‌ها:
  - این ماژول هیچ وابستگی سنگینی ندارد و از قوانین ساده/RegEx استفاده می‌کند.
  - نرمال‌سازی فارسی/عربی با perception.normalize_text (در صورت موجود بودن) انجام می‌شود.
  - اعتبارسنجی نهایی آرگومان‌ها در toolhub.verify صورت می‌گیرد.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import re
import os
import json
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    yaml = None  # type: ignore

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    joblib = None  # type: ignore

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

# نگاشت ارقام فارسی/عربی و نمادهای متداول به ASCII
_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _fa_to_ascii_math(text: str) -> str:
    if not text:
        return ""
    out = text.translate(_FA_DIGITS).translate(_AR_DIGITS)
    out = (
        out.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    return out


# قوانین آموخته‌شده (RULE/CLARIFY) — اختیاری
_LEARNED_RULES = []
if yaml is not None:
    try:
        rules_path = os.path.join("config", "learned_rules.yaml")
        if os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            raw_rules = list((data or {}).get("rules", []) or [])
            normalised = []
            for item in raw_rules:
                intent = str((item or {}).get("intent") or "").strip()
                patterns: list[str] = []
                for pat in list((item or {}).get("patterns", []) or []):
                    pat_norm = normalize_text(str(pat)) if pat else ""
                    if pat_norm:
                        patterns.append(pat_norm)
                if intent and patterns:
                    normalised.append({"intent": intent, "patterns": patterns})
            _LEARNED_RULES = normalised
    except Exception:
        _LEARNED_RULES = []


# کلاس‌بند نیت — اختیاری
_INTENT_CLF = None
if joblib is not None:
    try:
        clf_path = os.path.join("models", "intent_clf.joblib")
        if os.path.exists(clf_path):
            _INTENT_CLF = joblib.load(clf_path)
    except Exception:
        _INTENT_CLF = None


def _load_demo_inputs() -> list[str]:
    path = os.path.join("data", "demo_memory.jsonl")
    if not os.path.exists(path):
        return []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    inputs: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        text = normalize_text(str(obj.get("input", ""))) if obj else ""
        if text:
            inputs.append(text)
    return inputs

_DEMO_INPUTS = _load_demo_inputs()


def _demo_similarity(text: str) -> float:
    if not _DEMO_INPUTS:
        return 0.0
    tokens_q = set(text.split())
    if not tokens_q:
        return 0.0
    best = 0.0
    for item in _DEMO_INPUTS:
        tokens_t = set(item.split())
        if not tokens_t:
            continue
        inter = tokens_q & tokens_t
        union = tokens_q | tokens_t
        if not union:
            continue
        score = len(inter) / len(union)
        if score > best:
            best = score
    return best

def _has_any_math_symbol(text: str) -> bool:
    return any(ch in text for ch in "+-*/()×÷−–—")

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
    t_math = _fa_to_ascii_math(t)

    # 0) قوانین آموخته‌شده (RULE/CLARIFY)
    for rule in _LEARNED_RULES or []:
        intent = rule.get("intent") or ""
        if not intent:
            continue
        patterns = list(rule.get("patterns") or [])
        if any(p and p in t for p in patterns):
            return {"intent": intent, "confidence": 0.75, "args": {"raw": raw}}

    # 0.5) کلاس‌بند نیت (اختیاری)
    if _INTENT_CLF is not None:
        try:
            pred = _INTENT_CLF.predict([t])[0]
            pred_str = str(pred)
            if pred_str and pred_str not in {"unknown", "other"}:
                score = 0.8
                if hasattr(_INTENT_CLF, "predict_proba"):
                    try:
                        probs = _INTENT_CLF.predict_proba([t])[0]
                        if hasattr(probs, "__iter__"):
                            score = float(max(probs))
                    except Exception:
                        score = 0.8
                return {
                    "intent": pred_str,
                    "confidence": max(0.7, min(0.95, float(score))),
                    "args": {"raw": raw},
                }
        except Exception:
            pass

    # 1) سلام/احوالپرسی
    if _GREET_RE.search(t):
        return {"intent": "greeting", "confidence": 0.92, "args": {}}

    # 2) محاسبه‌ی عددی
    #    قواعد: وجود نشانه‌های ریاضی یا کلمات راهنما یا وجود یک عبارت قابل‌استخراج
    expr = _extract_expr(t_math)
    if expr or _has_any_math_symbol(t_math) or _COMPUTE_HINT_RE.search(t):
        if not expr:
            # ممکن است کاربر فقط بگوید "حاصل جمع ۷ و ۵" ولی ارقام لاتین نباشند
            return {"intent": "compute", "confidence": 0.70, "args": {"raw": raw}}
        return {"intent": "compute", "confidence": 0.85, "args": {"expr": expr, "raw": raw}}

    # 2.5) حافظه DEMO — اگر ورودی به نمونه‌های دمو شبیه بود
    if _demo_similarity(t) >= 0.3:
        return {"intent": "memory.reply", "confidence": 0.72, "args": {"raw": raw}}

    # 3) پیش‌فرض
    return {"intent": "smalltalk", "confidence": 0.60, "args": {"raw": raw}}

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
