# -*- coding: utf-8 -*-
"""
NOEMA • lang package (V0)
- لایه‌ی زبان: تحلیل نیت/آرگومان از متن + قالب‌بندی پاسخ خروجی.
- این ماژول فقط رابط‌های سطح‌بالا را اکسپورت می‌کند تا import ساده بماند.

استفاده:
    from lang import parse, detect_intent, load_style, format_reply, Style
    plan = parse("۲+۲؟")
    txt  = format_reply(intent=plan["intent"], outcome={"expr":"2+2","result":"4"}, style=load_style())
"""

from .parse import parse, detect_intent
from .format import load_style, format_reply, Style

__all__ = [
    "parse",
    "detect_intent",
    "load_style",
    "format_reply",
    "Style",
]
