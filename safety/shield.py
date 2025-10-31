# -*- coding: utf-8 -*-
"""
NOEMA • safety/shield.py — سپر ایمنی زمان اجرا (V0، سبک)

هدف:
  - قبل از «امتحان/اجرا»ی هر کنش (Action)، قواعد ایمنی را از safety/dsl بخواند
    و تصمیم بگیرد: allow / block / review. سپس:
       • در حالت allow → همان کنش عبور داده می‌شود.
       • در حالت review → می‌توانید کنش ایمن‌تری جایگزین کنید (مثل ask_clarify).
       • در حالت block → کنش رد می‌شود یا جایگزین ایمن برمی‌گردد.
  - همچنین می‌تواند لیست نامزدها را فیلتر کند.

وابستگی:
  - فقط safety/dsl (بدون کتابخانه‌ی خارجی)

رابط‌های اصلی:
  load_rules(path="config/safety.yaml") -> List[Rule]
  check_action(text, plan, action, state, rules, risk_base=0.0) -> Decision
  enforce(text, plan, action, state, rules, *, on_block="clarify") -> (Action, Decision)
  gate_candidates(text, plan, candidates, state, rules) -> (filtered, decisions)

یادداشت:
  - این ماژول چیزی را «اجرا» نمی‌کند؛ فقط اجازه/ممانعت را برمی‌گرداند.
  - در صورت نبود فایل پالیسی، همه‌چیز allow می‌شود (با risk_base).
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

# انواع world برای تایپ‌هینت (fallback کمینه)
try:
    from world import Action  # type: ignore
except Exception:
    from dataclasses import dataclass
    @dataclass
    class Action:
        kind: str
        name: str
        args: Dict[str, Any]

from .dsl import load_policies, evaluate, safe_decide_allow, Rule, Decision

# ───────────────────────────── بارگذاری قواعد ─────────────────────────────

def load_rules(path: str = "config/safety.yaml") -> List[Rule]:
    """قواعد ایمنی را از YAML/JSON بارگذاری می‌کند. اگر فایل نبود → لیست خالی."""
    return load_policies(path)

# ───────────────────────────── ارزیابی تک کنش ─────────────────────────────

def _make_context(
    text: str,
    plan: Dict[str, Any],
    action: Action,
    state: Optional[Dict[str, Any]] = None,
    risk_base: float = 0.0,
) -> Dict[str, Any]:
    return {
        "text": text or "",
        "plan": dict(plan or {}),
        "action": {"name": action.name, "kind": action.kind, "args": dict(action.args or {})},
        "state": dict(state or {}),
        "risk_base": float(risk_base or 0.0),
    }

def check_action(
    text: str,
    plan: Dict[str, Any],
    action: Action,
    state: Optional[Dict[str, Any]],
    rules: List[Rule],
    *,
    risk_base: float = 0.0,
) -> Decision:
    """
    خروجی: Decision(effect, matched, risk, reasons, details)
    """
    ctx = _make_context(text, plan, action, state, risk_base=risk_base)
    return evaluate(ctx, rules)

# ───────────────────────────── اعمال سپر ─────────────────────────────

def enforce(
    text: str,
    plan: Dict[str, Any],
    action: Action,
    state: Optional[Dict[str, Any]],
    rules: List[Rule],
    *,
    on_block: str = "clarify",     # "clarify" | "drop"
    risk_base: float = 0.0,
) -> Tuple[Action, Decision]:
    """
    تصمیم ایمنی را اعمال می‌کند:
      - allow  → همان action
      - review → اگر on_block="clarify" → ask_clarify جایگزین می‌شود
      - block  → بسته به on_block: clarify یا حذف (drop=برگرداندن ask_clarify با یادداشت)
    """
    dec = check_action(text, plan, action, state, rules, risk_base=risk_base)
    if dec.effect == "allow":
        return action, dec

    # در review/block: اگر سیاست «توضیح» فعال است، یک clarify جایگزین بده
    if on_block == "clarify":
        hint = "short" if plan.get("intent","unknown") == "unknown" else "detail"
        alt = Action(kind="policy", name="ask_clarify", args={"hint": hint, "reason": ", ".join(dec.matched)})
        return alt, dec

    # در حالت drop: همان action را برنگردانیم (اما برای سازگاری چیزی ایمن بدهیم)
    alt = Action(kind="policy", name="ask_clarify", args={"reason": ", ".join(dec.matched)})
    return alt, dec

# ───────────────────────────── فیلتر نامزدها ─────────────────────────────

def gate_candidates(
    text: str,
    plan: Dict[str, Any],
    candidates: List[Action],
    state: Optional[Dict[str, Any]],
    rules: List[Rule],
    *,
    risk_base: float = 0.0,
) -> Tuple[List[Action], List[Tuple[Action, Decision]]]:
    """
    لیست نامزدها را با قواعد ایمنی فیلتر می‌کند.
    خروجی:
      - filtered: فقط allow + (در صورت نبود هیچ allow) یک clarify
      - decisions: [(action, decision), ...] برای دیباگ
    """
    decisions: List[Tuple[Action, Decision]] = []
    safe_list: List[Action] = []

    for a in candidates:
        dec = check_action(text, plan, a, state, rules, risk_base=risk_base)
        decisions.append((a, dec))
        if dec.effect == "allow":
            safe_list.append(a)

    # اگر همه مسدود شدند، یک clarify ایمن اضافه کن
    if not safe_list:
        safe_list = [Action(kind="policy", name="ask_clarify", args={"reason": "no_safe_candidate"})]

    # حذف تکراری‌ها و بازگشت
    out: List[Action] = []
    seen = set()
    for a in safe_list:
        key = (a.kind, a.name, tuple(sorted((a.args or {}).items())))
        if key not in seen:
            out.append(a)
            seen.add(key)

    return out, decisions

# ───────────────────────────── خلاصه‌ی دلیل ─────────────────────────────

def reason_text(dec: Decision) -> str:
    """
    یک رشته‌ی کوتاه از دلایل تصمیم می‌سازد (برای لاگ یا پیام مربی).
    """
    prefix = {"allow":"✅", "review":"⚠️", "block":"⛔️"}.get(dec.effect, "")
    msg = " / ".join(dec.reasons) if dec.reasons else dec.effect
    return f"{prefix} {msg} [risk={dec.risk:.2f}, matched={','.join(dec.matched) or '-'}]"

# ───────────────────────────── تست مستقیم ─────────────────────────────

if __name__ == "__main__":
    # قواعد نمونه را از dsl لود کنید (اگر فایل نبود، خالی می‌آید)
    rules = load_rules()  # config/safety.yaml
    # چند کنش نمونه
    user_text = "جواب 7*(5-2) رو بگو"
    plan = {"intent": "compute", "args": {"raw": user_text}}
    state = {"u": 0.2, "conf": 0.8}

    a_calc = Action(kind="tool", name="invoke_calc", args={"expr": "7*(5-2)"})
    a_shell = Action(kind="tool", name="run_shell", args={"cmd": "rm -rf /"})
    a_clr = Action(kind="policy", name="ask_clarify", args={})

    for a in [a_calc, a_shell, a_clr]:
        dec = check_action(user_text, plan, a, state, rules, risk_base=0.05)
        print(a.name, "→", reason_text(dec))

    # فیلتر کردن نامزدها
    filtered, decisions = gate_candidates(user_text, plan, [a_calc, a_shell, a_clr], state, rules)
    print("filtered:", [f"{a.kind}:{a.name}" for a in filtered])
