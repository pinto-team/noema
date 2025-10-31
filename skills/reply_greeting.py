# -*- coding: utf-8 -*-
"""
NOEMA • skills/reply_greeting.py — مهارت «پاسخ سلام» (V0)

هدف:
  - یک پاسخ کوتاه، مودبانه/دوستانه به پیام‌های سلام تولید می‌کند.
  - خروجی هم به‌صورت outcome ساخت‌یافته و هم text_out آماده‌ی نمایش برمی‌گردد
    تا با lang.format سازگار باشد.

قرارداد run():
    run(
        user_text: str = "",
        *,
        plan: dict | None = None,      # مثلا {"intent":"greeting", ...}
        style: "lang.format.Style" | None = None,
        extras: dict | None = None,    # داده‌های اضافی اختیاری برای لاگ
        **kwargs
    ) -> dict

خروجی dict شامل کلیدهای زیر است:
    {
      "intent": "greeting",
      "outcome": {"variant": "default"},
      "text_out": "سلام! خوش اومدی.",
      "meta": {
         "confidence": 0.9,
         "u": 0.1,
         "r_total": 0.0,
         "risk": 0.0
      },
      "extras": {...}  # عبوری
    }

یادداشت:
  - confidence/u در این مهارت ثابت و محافظه‌کارانه انتخاب شده‌اند؛
    اگر self-model در لایه‌ی بالاتر وجود دارد، همان‌ها جایگزین می‌شوند.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

try:
    from lang import load_style, format_reply, Style  # type: ignore
except Exception:
    # fallback سبک در صورت نبود پکیج lang
    class Style:
        def __init__(self, tone: str = "friendly", formal: bool = False, prefix_emoji: bool = False):
            self.tone = tone
            self.formal = formal
            self.prefix_emoji = prefix_emoji
            self.max_len = 500
            self.show_confidence = False

    def load_style(path: str = "config/meta.yaml") -> Style:
        return Style()

    def format_reply(*, intent: str, outcome: Dict[str, Any] | None = None, style: Optional[Style] = None, meta: Optional[Dict[str, Any]] = None) -> str:
        style = style or Style()
        if style.formal:
            base = "درود بر شما."
        else:
            base = "سلام! خوش اومدی."
        return base

# ---------------------------------------------------------------------

_DEFAULT_META = {
    "confidence": 0.90,
    "u": 0.10,
    "r_total": 0.0,
    "risk": 0.0,
}

def run(
    user_text: str = "",
    *,
    plan: Optional[Dict[str, Any]] = None,
    style: Optional[Style] = None,
    extras: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    تولید پاسخ سلام.
    """
    style = style or load_style()
    outcome = {"variant": "default"}
    meta = dict(_DEFAULT_META)

    # اگر در plan چیزی درباره‌ی لحن آمده باشد، سبک را کمی تنظیم کن
    tone = (plan or {}).get("tone")
    if isinstance(tone, str):
        tone = tone.strip().lower()
        if tone in ("formal", "official"):
            style.formal = True
            style.prefix_emoji = False
        elif tone in ("friendly", "casual"):
            style.formal = False

    text_out = format_reply(intent="greeting", outcome=outcome, style=style, meta=meta)

    return {
        "intent": "greeting",
        "outcome": outcome,
        "text_out": text_out,
        "meta": meta,
        "extras": dict(extras or {}),
    }

# ---------------------------------------------------------------------

if __name__ == "__main__":
    print(run()["text_out"])
    print(run(plan={"tone":"formal"})["text_out"])
