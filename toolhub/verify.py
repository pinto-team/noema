# -*- coding: utf-8 -*-
"""
NOEMA • toolhub/verify.py — Lightweight argument contract checker (V0)

Goal
----
- Validate tool arguments against a simple contract:
  * allowed keys
  * required vs optional
  * types (str/int/float/bool/number/dict/list and list[T])
  * constraints: regex (for str), choices, min/max (numbers or list length), max_len (str)

API
---
verify_args(allowed: Dict[str, Any], provided: Dict[str, Any]) -> None
filter_allowed_kwargs(allowed: Dict[str, Any], provided: Dict[str, Any]) -> Dict[str, Any]

Behavior
--------
- Raises ValueError on unknown keys, missing required args, or invalid types/constraints.
- No coercion is performed (V0).
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import re

_SIMPLE_TYPES = {"str", "int", "float", "bool", "number", "dict", "list"}
_LIST_T_RE = re.compile(r"^list\[(?P<inner>str|int|float|bool|number|dict)\]$")

def _typeof(v: Any) -> str:
    if isinstance(v, bool):   # bool is a subclass of int
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
    return isinstance(v, (int, float)) and not isinstance(v, bool)

def _parse_type_spec(spec: Any) -> Tuple[str, Optional[str]]:
    if isinstance(spec, dict):
        spec = spec.get("type", "str")
    if not isinstance(spec, str):
        raise ValueError("type spec must be string or dict with 'type'")
    m = _LIST_T_RE.match(spec.strip())
    if m:
        return "list", m.group("inner")
    t = spec.strip().lower()
    if t not in _SIMPLE_TYPES and not _LIST_T_RE.match(t) and t != "number":
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
        rx = rule.get("regex")
        if isinstance(rx, str):
            if not re.fullmatch(rx, value or ""):
                raise ValueError("string does not match regex")
        mx = rule.get("max_len")
        if isinstance(mx, int) and mx > 0 and len(value) > mx:
            raise ValueError(f"string too long: {len(value)} > {mx}")
    elif base == "dict":
        if t != "dict":
            raise ValueError(f"expected dict, got {t}")
    else:
        raise ValueError(f"unsupported base type: {base}")

    if "choices" in rule:
        choices = rule["choices"]
        if isinstance(choices, (list, tuple, set)):
            if value not in choices:
                raise ValueError(f"value not in choices: {value}")

    if base in {"int", "float", "number"}:
        if "min" in rule and _is_number_ok(rule["min"]) and float(value) < float(rule["min"]):
            raise ValueError(f"value < min: {value} < {rule['min']}")
        if "max" in rule and _is_number_ok(rule["max"]) and float(value) > float(rule["max"]):
            raise ValueError(f"value > max: {value} > {rule['max']}")

def _validate_list(value: Any, inner: Optional[str], rule: Dict[str, Any]) -> None:
    if _typeof(value) != "list":
        raise ValueError(f"expected list, got {_typeof(value)}")
    if "min" in rule and isinstance(rule["min"], int) and len(value) < int(rule["min"]):
        raise ValueError(f"list length < min: {len(value)} < {rule['min']}")
    if "max" in rule and isinstance(rule["max"], int) and len(value) > int(rule["max"]):
        raise ValueError(f"list length > max: {len(value)} > {rule['max']}")
    if inner is None:
        return
    for x in value:
        _validate_scalar(x, "number" if inner == "number" else inner, {})

def verify_args(allowed: Dict[str, Any], provided: Dict[str, Any]) -> None:
    allowed = dict(allowed or {})
    provided = dict(provided or {})

    unknown = [k for k in provided.keys() if k not in allowed.keys()]
    if unknown:
        raise ValueError(f"unknown argument(s): {', '.join(unknown)}")

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
    allowed = dict(allowed or {})
    provided = dict(provided or {})
    return {k: v for k, v in provided.items() if k in allowed}

if __name__ == "__main__":
    allowed = {"expr": {"type": "str", "regex": r"^[0-9+\-*/() \t]+$", "max_len": 256}}
    verify_args(allowed, {"expr": "12*(3+1)-5"})
    try:
        verify_args(allowed, {"expr": "__import__('os')"})
    except Exception as e:
        print("rejected as expected:", e)
