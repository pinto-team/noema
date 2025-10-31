# -*- coding: utf-8 -*-
"""
NOEMA • toolhub/registry.py — رجیستری ابزارها (V0 سبک و فایل‌محور)

هدف:
  - یک لایه‌ی ساده برای تعریف/ثبت/لیست‌کردن ابزارهای در دسترس نوما.
  - سازگار با control/candidates.py (تابع list_safe_basics برای پیشنهاد ابزارهای امن).
  - قابلیت بارگذاری از YAML (config/tools.yaml) و بایند کردن تابع اجرایی (اختیاری).

تعاریف:
  • ToolSpec: فراابزار (متادیتا) شامل نام، نوع، سطح ایمنی، آرگومان‌های مجاز، برچسب‌ها…
  • ToolRegistry: نگهدارنده‌ی ToolSpecها + (اختیاری) بایند تابع اجرایی برای هر ابزار.

یادداشت:
  - این ماژول ابزار را «اجرا» نمی‌کند مگر آن‌که دستی bind شده باشد (fn).
  - پیش از اجرا می‌تواند از toolhub.verify برای بررسی ساده‌ی آرگومان‌ها استفاده کند.
  - اگر config/tools.yaml وجود داشته باشد، می‌توانید با load_from_yaml آن را بارگذاری کنید.

API خلاصه:
    reg = ToolRegistry()
    reg.register(ToolSpec(name="invoke_calc", safety="safe", tags=["basic"]))
    reg.bind("invoke_calc", my_calc_fn)
    names = reg.list_safe_basics()         # → ["invoke_calc", ...]
    out = reg.invoke("invoke_calc", expr="2+2")   # اگر بایند شده باشد

ساختار YAML (نمونه):
    tools:
      - name: invoke_calc
        kind: tool
        safety: safe
        desc: ماشین‌حساب چهاربعدی
        tags: [basic]
        allowed_args: { expr: "str" }
        cost: 0.05
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import json

# ───────────────────────────── ToolSpec ─────────────────────────────

@dataclass
class ToolSpec:
    name: str
    kind: str = "tool"                  # "tool" | "service" | ...
    desc: str = ""
    safety: str = "safe"                # "safe" | "guarded" | "high_risk"
    tags: List[str] = field(default_factory=list)
    allowed_args: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    enabled: bool = True

    # بایند اجرایی (در فایل ذخیره نمی‌شود)
    _fn: Optional[Callable[..., Any]] = field(default=None, repr=False, compare=False)

    # ویژگی‌های کمکی
    @property
    def is_safe_basic(self) -> bool:
        return self.enabled and (self.safety == "safe") and ("basic" in self.tags or True)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_fn", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolSpec":
        data = dict(d)
        data.pop("_fn", None)
        # هم‌سطح‌سازی جزئی
        data["name"] = str(data.get("name"))
        data["kind"] = str(data.get("kind", "tool"))
        data["desc"] = str(data.get("desc", ""))
        data["safety"] = str(data.get("safety", "safe"))
        data["tags"] = list(data.get("tags", []) or [])
        data["allowed_args"] = dict(data.get("allowed_args", {}) or {})
        data["cost"] = float(data.get("cost", 0.0) or 0.0)
        data["enabled"] = bool(data.get("enabled", True))
        return cls(**data)

# ───────────────────────────── Registry ─────────────────────────────

class ToolRegistry:
    """
    نگهدارنده‌ی ToolSpec ها + بایندهای اجرایی اختیاری.
    """

    def __init__(self):
        self._specs: Dict[str, ToolSpec] = {}

    # ثبت / حذف
    def register(self, spec: ToolSpec, fn: Optional[Callable[..., Any]] = None) -> None:
        s = spec
        if fn is not None:
            s._fn = fn
        self._specs[s.name] = s

    def remove(self, name: str) -> None:
        self._specs.pop(name, None)

    # بازیابی
    def has(self, name: str) -> bool:
        return name in self._specs

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def bind(self, name: str, fn: Callable[..., Any]) -> None:
        if name not in self._specs:
            # اگر قبلاً نبود، یک spec حداقلی بسازیم
            self._specs[name] = ToolSpec(name=name, desc="(auto-registered)", tags=["basic"])
        self._specs[name]._fn = fn

    # فهرست‌ها
    def list_all(self) -> List[str]:
        return sorted([n for n, s in self._specs.items() if s.enabled])

    def list_safe_basics(self) -> List[str]:
        """
        برای استفاده‌ی candidates.generate وقتی intent ناشناخته است.
        ترجیح: ابزارهای enabled و safety="safe" با تگ "basic".
        اگر خالی شد اما invoke_calc وجود داشت، آن را اضافه کن.
        """
        names = [n for n, s in self._specs.items() if s.enabled and s.safety == "safe"]
        # اگر برچسب basic وجود دارد، اولویت بده
        basics = [n for n, s in self._specs.items() if s.enabled and s.safety == "safe" and ("basic" in s.tags)]
        out = basics or names
        out = sorted(set(out))
        if "invoke_calc" in self._specs and "invoke_calc" not in out:
            out.append("invoke_calc")
        return out

    def suggest_for_intent(self, intent: str) -> List[str]:
        """
        نگاشت ساده intent→ابزارهای مرتبط (قابل‌گسترش).
        """
        intent = (intent or "").strip().lower()
        if intent == "compute":
            return [n for n in self.list_all() if n == "invoke_calc"]
        # TODO: برای intentهای دیگر توسعه دهید (search, browse, summarize, ...)
        return self.list_safe_basics()

    # اجرا (اختیاری)
    def invoke(self, name: str, **kwargs) -> Any:
        """
        اگر ابزار بایند شده باشد اجرا می‌کند؛ در غیر این صورت خطا می‌دهد.
        پیش از اجرا، verify_args ساده از toolhub.verify را صدا می‌زند (اگر در دسترس باشد).
        """
        spec = self._specs.get(name)
        if spec is None or not spec.enabled:
            raise KeyError(f"tool not found or disabled: {name}")

        # بررسی ساده‌ی آرگومان‌ها
        try:
            from toolhub.verify import verify_args  # type: ignore
            verify_args(spec.allowed_args, kwargs)
        except Exception:
            # اگر verify در دسترس نبود یا خطای جزئی داشت، ادامه می‌دهیم (V0 ملایم)
            pass

        if spec._fn is None:
            raise NotImplementedError(f"tool '{name}' has no bound function.")
        return spec._fn(**kwargs)

    # I/O پیکربندی
    def load_from_yaml(self, path: str | Path) -> int:
        """
        فایل YAML/JSON-شکل را می‌خواند و ToolSpecها را ثبت می‌کند.
        خروجی: تعداد ابزارهای بارگذاری‌شده.
        """
        cfg = _load_yaml(path)
        tools = cfg.get("tools", [])
        n = 0
        for t in tools:
            try:
                spec = ToolSpec.from_dict(t)
                # اگر نام تکراری بود، overwrite
                self._specs[spec.name] = spec
                n += 1
            except Exception:
                continue
        return n

    def save_to_json(self, path: str | Path = "data/tools.json") -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for n in self.list_all():
                f.write(json.dumps(self._specs[n].to_dict(), ensure_ascii=False) + "\n")
        return p

# ───────────────────────────── YAML loader ─────────────────────────────

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    # تلاش برای YAML؛ اگر نبود، JSON
    try:
        import yaml  # type: ignore
        obj = yaml.safe_load(text) or {}
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}

# ───────────────────────────── نمونه‌ی استفاده ─────────────────────────────

if __name__ == "__main__":
    reg = ToolRegistry()
    # ثبت حداقلی invoke_calc
    reg.register(ToolSpec(
        name="invoke_calc",
        kind="tool",
        desc="ماشین‌حساب امن چهارنویی",
        safety="safe",
        tags=["basic"],
        allowed_args={"expr": "str"},
        cost=0.05,
    ))

    # یک بایند ساده برای تست
    def _calc(expr: str) -> str:
        # ارزیابی بسیار محدود: فقط 0-9 + - * / ( )
        import re
        if not re.fullmatch(r"[0-9+\-*/() \t]+", expr):
            raise ValueError("invalid expr")
        return str(eval(expr, {"__builtins__": {}}, {}))
    reg.bind("invoke_calc", _calc)

    print("safe basics:", reg.list_safe_basics())
    print("suggest for compute:", reg.suggest_for_intent("compute"))
    print("invoke calc:", reg.invoke("invoke_calc", expr="2+2"))
