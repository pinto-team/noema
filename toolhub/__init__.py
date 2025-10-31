# -*- coding: utf-8 -*-
"""
NOEMA • toolhub package (V0)
- رجیستری ابزارها + اعتبارسنجی آرگومان‌ها + لودر پیکربندی.
- وابستگی حداقلی؛ بدون وابستگی به skills تا از چرخه‌ی import جلوگیری شود.

استفاده‌ی سریع:
    from toolhub import load_registry
    reg = load_registry("config/tools.yaml")      # اگر فایل نبود، خالی می‌ماند
    out = reg.invoke("invoke_calc", expr="2+2")   # اگر bind_calc=True باشد

نکته:
- load_registry به‌صورت اختیاری یک ماشین‌حساب امن مینیمال را ثبت/بایند می‌کند
  تا مسیر «ناشناخته → ابزار امن» کار کند. برای حذف آن، bind_calc=False بدهید.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import re

from .registry import ToolRegistry, ToolSpec
from .verify import verify_args, filter_allowed_kwargs

__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "verify_args",
    "filter_allowed_kwargs",
    "load_registry",
]

# ---------------------------------------------------------------------

def _safe_calc(expr: str) -> str:
    """
    ارزیاب بسیار محدودِ عبارات عددی: فقط 0-9، + - * / ( ) و فاصله.
    - هیچ نام/توابع/ساختار پایتونی اجازه ندارد.
    """
    if not isinstance(expr, str):
        raise ValueError("expr must be str")
    if not re.fullmatch(r"[0-9+\-*/() \t]+", expr):
        raise ValueError("invalid characters in expression")
    # eval ایمن با محیط خالی
    return str(eval(expr, {"__builtins__": {}}, {}))

def load_registry(config_path: str | Path = "config/tools.yaml", *, bind_calc: bool = True) -> ToolRegistry:
    """
    رجیستری را می‌سازد، از YAML (اگر وجود داشت) لود می‌کند، و به‌صورت اختیاری
    ماشین‌حساب امن را ثبت/بایند می‌کند.
    """
    reg = ToolRegistry()
    # 1) لود از فایل (اگر موجود)
    p = Path(config_path)
    if p.exists():
        reg.load_from_yaml(p)

    # 2) بایند ماشین‌حساب امن (اختیاری)
    if bind_calc:
        if not reg.has("invoke_calc"):
            reg.register(ToolSpec(
                name="invoke_calc",
                kind="tool",
                desc="ماشین‌حساب امن (چهار عمل اصلی)",
                safety="safe",
                tags=["basic"],
                allowed_args={
                    "expr": {"type": "str", "regex": r"^[0-9+\-*/() \t]+$", "max_len": 256}
                },
                cost=0.05,
            ))
        reg.bind("invoke_calc", _safe_calc)

    return reg
