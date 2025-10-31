# -*- coding: utf-8 -*-
"""
NOEMA • safety package (V0)
- DSL قواعد ایمنی + سپر زمان‌اجرا برای غربال/اجرا.

استفاده‌ی سریع:
    from safety import load_rules, check_action, enforce, gate_candidates, reason_text

    rules = load_rules("config/safety.yaml")
    a_safe, decision = enforce(user_text, plan, action, state, rules)
    print(reason_text(decision))

یادداشت:
- اگر فایل config/safety.yaml موجود نباشد، لیست قواعد خالی می‌آید و رفتار پیش‌فرض allow است.
- این پکیج چیزی را «اجرا» نمی‌کند؛ فقط اجازه/ممانعت و دلایل را برمی‌گرداند.
"""

from __future__ import annotations

# باز-صادر کردن مؤلفه‌های اصلی
from .dsl import (
    load_policies as load_rules,
    evaluate,
    safe_decide_allow,
    Rule,
    Decision,
)

from .shield import (
    check_action,
    enforce,
    gate_candidates,
    reason_text,
)

__all__ = [
    # DSL
    "load_rules",
    "evaluate",
    "safe_decide_allow",
    "Rule",
    "Decision",
    # Shield
    "check_action",
    "enforce",
    "gate_candidates",
    "reason_text",
]

# اجرای کوتاه برای اطمینان از واردشدن بدون خطا
if __name__ == "__main__":
    rules = load_rules()  # اگر نبود، خالی
    print(f"safety loaded — {len(rules)} rule(s)")
