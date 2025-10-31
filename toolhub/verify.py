# -*- coding: utf-8 -*-
"""
NOEMA • toolhub/verify.py — اعتبارسنج سبکِ آرگومان‌های ابزار (V0، فقط stdlib)

هدف:
  - قبل از اجرای هر ابزار، آرگومان‌ها را با «قرارداد ساده» بررسی کند.
  - با toolhub/registry.py سازگار است (verify_args در زمان invoke صدا زده می‌شود).

قرارداد توصیف آرگومان‌ها (allowed_args):
  1) رشته‌ی نوع:
       "str" | "int" | "float" | "bool" | "number" | "dict" | "list" |
       "list[str]" | "list[int]" | "list[float]" | "list[dict]" | ...
  2) شیء دیکشنری با قیود:
       {
         "type": "str" | "int" | "float" | "bool" | "number" | "dict" | "list[...]",
         "optional": true/false,            # پیش‌فرض false
         "regex": "^[0-9+\\-*/() \\t]+$",   # فقط برای str
         "choices": ["a","b",...],          # فقط برای str/int/float/bool/number
         "min": 0, "max": 100,              # برای int/float/number یا طول لیست
         "max_len": 512                     # برای str
       }

API:
  verify_args(allowed: Dict[str, Any], provided: Dict[str, Any]) -> None
  filter_allowed_kwargs(allowed: Dict[str, Any], provided: Dict[str, Any]) -> Dict[str, Any]

رفتار:
  - اگر کلید ناشناخته یا نوع نامعتبر باشد → ValueError
  - اگر کلید لازم نیامده باشد → ValueError
  - در V0 هیچ تبدیل نوع (coercion) انجام نمی‌شود؛ تنها بررسی می‌کند.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import re
import json

# ───────────────────────────── ابزار نوع ─────────────────────────────

_SIMPLE_TYPES = {"str", "int", "float", "bool", "number", "dict", "list"}

_LIST_T_RE = re.compile(r"^list\[(?P<inner>str|int|float|bool|number|dict)\]$")

def _typeof(v: Any) -> str:
    if isinstance(v, bool):   # توجه: bool زیرکلاس int است
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, (list, tuple)):
        return "list"
    return type(v).__name__

def _is_number_ok(v: Any) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, bool))

def _parse_type_spec(spec: Any) -> Tuple[str, Optional[str]]:
    """
    spec "list[T]" → ("list", "T")   |   "str" → ("str", None)
    اگر dict بود، از فیلد type می‌خواند.
    """
    if isinstance(spec, dict):
        spec = spec.get("type", "str")
    if not isinstance(spec, str):
        raise ValueError("type spec must be string or dict with 'type'")
    m = _LIST_T_RE.match(spec.strip())
    if m:
        return "list", m.group("inner")
    t = spec.strip().lower()
    if t not in _SIMPLE_TYPES and not _LIST_T_RE.match(t):
        # برای "number" هم اجازه می‌دهیم
        if t not in {"number"}:
            raise ValueError(f"unsupported type spec: {t}")
    return t, None

def _validate_scalar(value: Any, base: str, rule: Dict[str, Any]) -> None:
    t = _typeof(value)
    if base == "number":
        if not _is_number_ok(value):
            raise ValueError(f"expected number, got {t}")
    elif base == "bool":
        if t != "bool":
            raise ValueError(f"expected bool, got {t}")
    elif base == "int":
        if t != "int":
            raise ValueError(f"expected int, got {t}")
    elif base == "float":
        if t not in ("float", "int") or isinstance(value, bool):
            raise ValueError(f"expected float, got {t}")
    elif base == "str":
        if t != "str":
            raise ValueError(f"expected str, got {t}")
        # regex
        rx = rule.get("regex")
        if isinstance(rx, str):
            if not re.fullmatch(rx, value or ""):
                raise ValueError("string does not match regex")
        # max_len
        mx = rule.get("max_len")
        if isinstance(mx, int) and mx > 0 and len(value) > mx:
            raise ValueError(f"string too long: {len(value)} > {mx}")
    elif base == "dict":
        if t != "dict":
            raise ValueError(f"expected dict, got {t}")
    else:
        raise ValueError(f"unsupported base type: {base}")

    # choices
    if "choices" in rule:
        choices = rule["choices"]
        if isinstance(choices, (list, tuple, set)):
            if value not in choices:
                raise ValueError(f"value not in choices: {value}")

    # min/max (برای عددی‌ها)
    if base in {"int", "float", "number"}:
        if "min" in rule and _is_number_ok(rule["min"]) and float(value) < float(rule["min"]):
            raise ValueError(f"value < min: {value} < {rule['min']}")
        if "max" in rule and _is_number_ok(rule["max"]) and float(value) > float(rule["max"]):
            raise ValueError(f"value > max: {value} > {rule['max']}")

def _validate_list(value: Any, inner: Optional[str], rule: Dict[str, Any]) -> None:
    if _typeof(value) != "list":
        raise ValueError(f"expected list, got {_typeof(value)}")
    # min/max روی طول
    if "min" in rule and isinstance(rule["min"], int) and len(value) < int(rule["min"]):
        raise ValueError(f"list length < min: {len(value)} < {rule['min']}")
    if "max" in rule and isinstance(rule["max"], int) and len(value) > int(rule["max"]):
        raise ValueError(f"list length > max: {len(value)} > {rule['max']}")
    # نوع اعضا
    if inner is None:
        return
    for i, x in enumerate(value):
        _validate_scalar(x, "number" if inner == "number" else inner, {})  # بدون قید اضافی روی اعضا

# ───────────────────────────── API اصلی ─────────────────────────────

def verify_args(allowed: Dict[str, Any], provided: Dict[str, Any]) -> None:
    """
    بررسی می‌کند که:
      - کلید ناشناخته وجود نداشته باشد
      - کلیدهای لازم (optional=false) آمده باشند
      - انواع و قیود (regex/choices/len/min/max) رعایت شده باشد
    خطاها به صورت ValueError بالا می‌روند.
    """
    allowed = dict(allowed or {})
    provided = dict(provided or {})

    # کلید ناشناخته
    unknown = [k for k in provided.keys() if k not in allowed.keys()]
    if unknown:
        raise ValueError(f"unknown argument(s): {', '.join(unknown)}")

    # هر کلید مجاز را بررسی کن
    for name, spec in allowed.items():
        rule = spec if isinstance(spec, dict) else {"type": spec}
        optional = bool(rule.get("optional", False))
        if name not in provided:
            if not optional:
                raise ValueError(f"missing required argument: {name}")
            else:
                continue

        val = provided[name]
        base, inner = _parse_type_spec(rule)

        if base == "list":
            _validate_list(val, inner, rule)
        else:
            _validate_scalar(val, base, rule)

def filter_allowed_kwargs(allowed: Dict[str, Any], provided: Dict[str, Any]) -> Dict[str, Any]:
    """
    فقط کلیدهای مجاز را عبور می‌دهد (برای ایمنی بیشتر هنگام فراخوانی توابع).
    اگر verify_args را قبلاً صدا زده باشید، این تابع صرفاً یک فیلتر کوچک است.
    """
    allowed = dict(allowed or {})
    provided = dict(provided or {})
    return {k: v for k, v in provided.items() if k in allowed}

# ───────────────────────────── نمونه‌ی استفاده ─────────────────────────────

if __name__ == "__main__":
    # مثال ۱: ماشین‌حساب
    allowed = {
        "expr": {
            "type": "str",
            "regex": r"^[0-9+\-*/() \t]+$",
            "max_len": 256,
        }
    }
    ok = {"expr": "12*(3+1) - 5"}
    bad = {"expr": "__import__('os').system('rm -rf /')"}  # باید رد شود

    try:
        verify_args(allowed, ok)
        print("✅ calc ok passed")
    except Exception as e:
        print("❌ calc ok failed:", e)

    try:
        verify_args(allowed, bad)
        print("❌ calc bad passed (should fail)")
    except Exception as e:
        print("✅ calc bad rejected:", e)

    # مثال ۲: لیست اعداد با کران
    allowed2 = {
        "values": {"type": "list[float]", "min": 1, "max": 4}
    }
    verify_args(allowed2, {"values": [0.1, 2, 3.14]})
    try:
        verify_args(allowed2, {"values": []})
    except Exception as e:
        print("✅ empty list rejected:", e)
