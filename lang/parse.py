# -*- coding: utf-8 -*-
"""
NOEMA • lang/parse.py — Lightweight intent parsing & argument extraction (V0, improved)

- Robust FA normalization (remove ZW chars, unifying Arabic/Farsi forms incl. آ→ا).
- Compute extraction handles Persian operator words only when numbers are on both sides.
- Intent order: learned RULE/CLARIFY → classifier → greeting → compute → demo-memory → smalltalk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
import json
import os
import re
import unicodedata

# ----------------------------- Optional deps -----------------------------
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

# ----------------------------- Zero-width & FA maps -----------------------------
_ZW_RE = re.compile(r"[\u200c\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_FA_CHARMAP = str.maketrans({
    "ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا", "آ": "ا", "ـ": ""
})

def _normalize_fa_text(t: str) -> str:
    """NFC + remove ZW + unify FA/AR letters + squeeze spaces + lowercase."""
    if not t:
        return ""
    t = unicodedata.normalize("NFC", t)
    t = _ZW_RE.sub("", t)
    t = t.translate(_FA_CHARMAP)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

# If perception.normalize_text exists, use it, else our FA-normalizer.
try:
    from perception import normalize_text as _perception_norm  # type: ignore
    def normalize_text(t: str) -> str:
        # Wrap to also remove ZW & unify FA chars even if perception provides a basic normalizer
        return _normalize_fa_text(_perception_norm(t))
except Exception:
    def normalize_text(t: str) -> str:
        return _normalize_fa_text(t)

# ----------------------------- Greetings (fallback, latin) -----------------------------
_GREET_WORDS = ["hi","hello","hey","hola","hallo","ciao","salut","bonjour","namaste","ola","hei"]
_GREET_RE = re.compile("|".join([re.escape(w) for w in _GREET_WORDS]), re.IGNORECASE)

# ----------------------------- Compute hints (EN + FA) -----------------------------
_COMPUTE_HINT_EN_RE = re.compile(
    r"(result|compute|calculate|sum|plus|minus|multiply|divide|answer|equals|calc)",
    re.IGNORECASE,
)

_COMPUTE_HINT_FA_RE = re.compile(
    r"(حاصل|محاسبه|نتیجه|چند\s*میشه|چنده|تقسیم\s*بر|تقسیم|ضرب\s*در|ضربدر|ضرب|"
    r"به\s*علاوه|جمع|منهای)",
    re.IGNORECASE,
)

# ----------------------------- Expression charset guards -----------------------------
_EXPR_RE = re.compile(r"[0-9+\-*/() \t]+")

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EXT_ARABIC_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Visual math symbols → ASCII
_VISUAL_MATH = {"×": "*", "÷": "/", "−": "-", "–": "-", "—": "-"}

def _strip_cf(t: str) -> str:
    return "".join(ch for ch in t if unicodedata.category(ch) != "Cf")

def _digits_to_ascii(t: str) -> str:
    return t.translate(_EXT_ARABIC_INDIC).translate(_ARABIC_INDIC)

# ----------------------------- FA operator mapping (number-bound) -----------------------------
# NOTE: only map operator words when numbers appear on both sides to avoid harming words like "درود/برنامه".
_NUM = r"([0-9]+)"  # after digits_to_ascii we look only for ASCII digits

_NUM_OP_PATTERNS: List[tuple[re.Pattern, str]] = [
    (re.compile(_NUM + r"\s*تقسیم\s*بر\s*" + _NUM, re.IGNORECASE), r"\1 / \2"),
    (re.compile(_NUM + r"\s*تقسیم\s*"    + _NUM, re.IGNORECASE),     r"\1 / \2"),
    (re.compile(_NUM + r"\s*ضرب\s*در\s*" + _NUM, re.IGNORECASE),     r"\1 * \2"),
    (re.compile(_NUM + r"\s*ضربدر\s*"    + _NUM, re.IGNORECASE),     r"\1 * \2"),
    (re.compile(_NUM + r"\s*ضرب\s*"      + _NUM, re.IGNORECASE),     r"\1 * \2"),
    (re.compile(_NUM + r"\s*در\s*"       + _NUM, re.IGNORECASE),     r"\1 * \2"),
    (re.compile(_NUM + r"\s*به\s*علاوه\s*"+ _NUM, re.IGNORECASE),    r"\1 + \2"),
    (re.compile(_NUM + r"\s*جمع\s*"      + _NUM, re.IGNORECASE),     r"\1 + \2"),
    (re.compile(_NUM + r"\s*منهای\s*"    + _NUM, re.IGNORECASE),     r"\1 - \2"),
    (re.compile(_NUM + r"\s*تا\s*"       + _NUM, re.IGNORECASE),     r"\1 * \2"),
]

def _map_fa_ops_when_number_bound(t: str) -> str:
    s = t
    for pat, rep in _NUM_OP_PATTERNS:
        s = pat.sub(rep, s)
    return s

def _visual_symbols_to_ascii(t: str) -> str:
    out = t
    for k, v in _VISUAL_MATH.items():
        out = out.replace(k, v)
    return out

def _to_ascii_math(text: str) -> str:
    """
    Turn arbitrary text into a math-friendly ASCII string, carefully:
    - remove ZW/Cf;
    - normalize FA letters (esp. آ→ا so RULEs match variations);
    - map FA/AR digits to ASCII;
    - map Persian operator words ONLY when numbers appear on both sides;
    - map visual math symbols to ASCII.
    """
    if not text:
        return ""
    t = _strip_cf(text)
    t = _normalize_fa_text(t)           # unify FA letters, collapse spaces, lower
    t = _digits_to_ascii(t)             # convert digits to ASCII
    t = _map_fa_ops_when_number_bound(t)
    t = _visual_symbols_to_ascii(t)
    return t

# ----------------------------- Learned rules (RULE/CLARIFY) -----------------------------
_LEARNED_RULES: List[Dict[str, Any]] = []
def _load_learned_rules() -> List[Dict[str, Any]]:
    if yaml is None:
        return []
    try:
        rules_path = os.path.join("config", "learned_rules.yaml")
        if not os.path.exists(rules_path):
            return []
        with open(rules_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        raw_rules = list((data or {}).get("rules", []) or [])
        norm_rules: List[Dict[str, Any]] = []
        for item in raw_rules:
            intent = str((item or {}).get("intent") or "").strip()
            patterns: List[str] = []
            for pat in list((item or {}).get("patterns", []) or []):
                pat_norm = normalize_text(str(pat)) if pat else ""
                if pat_norm:
                    patterns.append(pat_norm)
            if intent and patterns:
                norm_rules.append({"intent": intent, "patterns": patterns})
        return norm_rules
    except Exception:
        return []

_LEARNED_RULES = _load_learned_rules()

# ----------------------------- Optional intent classifier -----------------------------
_INTENT_CLF = None
if joblib is not None:
    try:
        clf_path = os.path.join("models", "intent_clf.joblib")
        if os.path.exists(clf_path):
            _INTENT_CLF = joblib.load(clf_path)
    except Exception:
        _INTENT_CLF = None

# ----------------------------- DEMO inputs for similarity -----------------------------
def _load_demo_inputs() -> List[str]:
    path = os.path.join("data", "demo_memory.jsonl")
    if not os.path.exists(path):
        return []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    inputs: List[str] = []
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
    """Very simple Jaccard similarity over whitespace tokens against DEMO inputs."""
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

# ----------------------------- Helpers -----------------------------
def _has_any_math_symbol(text: str) -> bool:
    # either ASCII math tokens or FA compute hints in the original text
    return any(ch in text for ch in "+-*/()") or bool(_COMPUTE_HINT_FA_RE.search(text))

def _extract_expr(text: str) -> Optional[str]:
    """
    Try to extract a safe ASCII expression from already-normalized math text.
    Accept only [0-9 + - * / ( )] to keep it safe.
    """
    # Find the longest plausible expression
    m = re.search(r"([0-9+\-*/() \t]{2,})", text)
    if not m:
        return None
    expr = (m.group(1) or "").strip()
    if not expr:
        return None
    if not _EXPR_RE.fullmatch(expr):
        return None
    return expr

# ----------------------------- Intent detection -----------------------------
def detect_intent(text: str) -> Dict[str, Any]:
    """
    Lightweight intent detection with optional learned artifacts.
    Returns: {"intent": "...", "confidence": 0.xx, "args": {...}}
    """
    raw = text or ""
    t = normalize_text(raw)       # FA-robust normalization for lexical matching
    t_math = _to_ascii_math(raw)  # math-oriented normalization from original raw

    # 0) Learned rules (RULE/CLARIFY) — exact lexical contains on normalized strings
    for rule in _LEARNED_RULES or []:
        intent = rule.get("intent") or ""
        if not intent:
            continue
        patterns = list(rule.get("patterns") or [])
        if any(p and p in t for p in patterns):
            return {"intent": intent, "confidence": 0.78, "args": {"raw": raw}}

    # 0.5) Optional intent classifier
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

    # 1) Greeting (fallback; english tokens)
    if _GREET_RE.search(t):
        return {"intent": "greeting", "confidence": 0.92, "args": {}}

    # 2) Arithmetic (handles FA operator words when numbers bound)
    expr = _extract_expr(t_math)
    if expr or _has_any_math_symbol(raw) or _COMPUTE_HINT_EN_RE.search(t):
        if not expr:
            # Hints present but no valid ASCII expr yet
            return {"intent": "compute", "confidence": 0.70, "args": {"raw": raw}}
        return {
            "intent": "compute",
            "confidence": 0.86,
            "args": {"expr": expr, "raw": raw},
        }

    # 2.5) DEMO memory similarity
    if _demo_similarity(t) >= 0.3:
        return {"intent": "memory.reply", "confidence": 0.74, "args": {"raw": raw}}

    # 3) Default smalltalk
    return {"intent": "smalltalk", "confidence": 0.60, "args": {"raw": raw}}

# ----------------------------- Public API -----------------------------
def parse(text: str, wm: Optional[Any] = None) -> Dict[str, Any]:
    """Public entry. `wm` reserved for future extensions."""
    return detect_intent(text)

# ----------------------------- Quick self-test -----------------------------
if __name__ == "__main__":
    samples = [
        "hi there!",
        "2+2?",
        "نتیجه ۷*(۵-۲)؟",
        "حاصل ۲ به علاوه ۲",
        "۹ تقسیم بر ۳",
        "3 ضربدر 4",
        "آفرین",
        "what's the weather like",
    ]
    for s in samples:
        p = parse(s)
        print(f"{s!r} -> {p}")
