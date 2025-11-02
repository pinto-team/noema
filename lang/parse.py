# -*- coding: utf-8 -*-
"""
NOEMA • lang/parse.py — Lightweight intent parsing & argument extraction (V0)

Purpose:
  - Turn raw user text into a simple "plan" for the controller.
  - Supported intents in V0: [greeting, compute, smalltalk, memory.reply] + learned RULE intents.
  - For compute, tries to extract a safe arithmetic expression.
  - If there is a similar DEMO example, surface memory.reply (to use memory skill).

Notes:
  - No heavy dependencies; relies on simple normalization, regex, and optional artifacts.
  - Optional artifacts: config/learned_rules.yaml, models/intent_clf.joblib, data/demo_memory.jsonl
  - Final argument validation is handled elsewhere (e.g., toolhub.verify).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
import json
import os
import re

# Optional deps
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

# Optional normalization from perception; fallback is language-agnostic
try:
    from perception import normalize_text  # type: ignore
except Exception:
    import unicodedata

    def normalize_text(t: str) -> str:
        if not t:
            return ""
        t = unicodedata.normalize("NFC", t)
        t = re.sub(r"\s+", " ", t).strip().lower()
        return t


# Minimal multilingual greeting tokens (latin-script only; fallback)
_GREET_WORDS = [
    "hi",
    "hello",
    "hey",
    "hola",
    "hallo",
    "ciao",
    "salut",
    "bonjour",
    "namaste",
    "ola",
    "hei",
]
_GREET_RE = re.compile("|".join([re.escape(w) for w in _GREET_WORDS]), re.IGNORECASE)

# Compute hint keywords (language-agnostic fallback; latin tokens)
_COMPUTE_HINT_RE = re.compile(
    r"(result|compute|calculate|sum|plus|minus|multiply|divide|answer|equals|calc)",
    re.IGNORECASE,
)

# Safe arithmetic expression pattern (aligned with tool validation)
_EXPR_RE = re.compile(r"[0-9+\-*/() \t]+")

# Map Arabic-Indic and Extended Arabic-Indic digits, and math-like symbols, to ASCII
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_EXT_ARABIC_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def _to_ascii_math(text: str) -> str:
    if not text:
        return ""
    out = text.translate(_EXT_ARABIC_INDIC).translate(_ARABIC_INDIC)
    out = (
        out.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
    )
    return out


# Learned rules (RULE/CLARIFY) — optional
_LEARNED_RULES: List[Dict[str, Any]] = []
if yaml is not None:
    try:
        rules_path = os.path.join("config", "learned_rules.yaml")
        if os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            raw_rules = list((data or {}).get("rules", []) or [])
            normalized: List[Dict[str, Any]] = []
            for item in raw_rules:
                intent = str((item or {}).get("intent") or "").strip()
                patterns: List[str] = []
                for pat in list((item or {}).get("patterns", []) or []):
                    pat_norm = normalize_text(str(pat)) if pat else ""
                    if pat_norm:
                        patterns.append(pat_norm)
                if intent and patterns:
                    normalized.append({"intent": intent, "patterns": patterns})
            _LEARNED_RULES = normalized
    except Exception:
        _LEARNED_RULES = []


# Optional intent classifier
_INTENT_CLF = None
if joblib is not None:
    try:
        clf_path = os.path.join("models", "intent_clf.joblib")
        if os.path.exists(clf_path):
            _INTENT_CLF = joblib.load(clf_path)
    except Exception:
        _INTENT_CLF = None


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


def _has_any_math_symbol(text: str) -> bool:
    return any(ch in text for ch in "+-*/()×÷−–—")


def _extract_expr(text: str) -> Optional[str]:
    m = re.search(r"([0-9+\-*/() \t]{2,})", text)
    if not m:
        return None
    expr = (m.group(1) or "").strip()
    if not expr:
        return None
    if not _EXPR_RE.fullmatch(expr):
        return None
    return expr


# ---------------------------- Intent detection ----------------------------

def detect_intent(text: str) -> Dict[str, Any]:
    """
    Lightweight intent detection with optional learned artifacts.
    Returns: {"intent": "...", "confidence": 0.xx, "args": {...}}
    """
    raw = text or ""
    t = normalize_text(raw)
    t_math = _to_ascii_math(t)

    # 0) Learned rules (RULE/CLARIFY)
    for rule in _LEARNED_RULES or []:
        intent = rule.get("intent") or ""
        if not intent:
            continue
        patterns = list(rule.get("patterns") or [])
        if any(p and p in t for p in patterns):
            return {"intent": intent, "confidence": 0.75, "args": {"raw": raw}}

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

    # 1) Greeting (fallback)
    if _GREET_RE.search(t):
        return {"intent": "greeting", "confidence": 0.92, "args": {}}

    # 2) Arithmetic
    expr = _extract_expr(t_math)
    if expr or _has_any_math_symbol(t_math) or _COMPUTE_HINT_RE.search(t):
        if not expr:
            # Symbols/hints present but no valid ASCII expr yet
            return {"intent": "compute", "confidence": 0.70, "args": {"raw": raw}}
        return {
            "intent": "compute",
            "confidence": 0.85,
            "args": {"expr": expr, "raw": raw},
        }

    # 2.5) DEMO memory similarity
    if _demo_similarity(t) >= 0.3:
        return {"intent": "memory.reply", "confidence": 0.72, "args": {"raw": raw}}

    # 3) Default
    return {"intent": "smalltalk", "confidence": 0.60, "args": {"raw": raw}}


# ---------------------------- Public API ----------------------------

def parse(text: str, wm: Optional[Any] = None) -> Dict[str, Any]:
    """
    Public entry. `wm` is reserved for future use (e.g., short-term context).
    """
    plan = detect_intent(text)
    return plan


# ---------------------------- Quick self-test ----------------------------

if __name__ == "__main__":
    samples = [
        "hi there!",
        "2+2?",
        "result of 7*(5-2)?",
        "what's the weather like",
    ]
    for s in samples:
        p = parse(s)
        print(f"{s!r} -> {p}")
