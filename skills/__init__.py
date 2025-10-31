# -*- coding: utf-8 -*-
"""
NOEMA • skills package (V0)
- بارگذاری «مهارت‌ها» از skills/manifest.yaml و اجرای آن‌ها به‌صورت پویا.
- طراحی سبک، بدون وابستگی سنگین؛ YAML اختیاری.

کارکردها:
    reg = load_skills("skills/manifest.yaml")
    out = reg.run("reply_greeting", user_text="سلام")

    # میان‌بُرها:
    reg = load_skills()                 # مسیر پیش‌فرض
    out = run_skill(reg, "reply_greeting", user_text="سلام")

یادداشت‌ها:
- برای مهارت‌های نوع tool (مثل invoke_calc) ممکن است entry خالی باشد و
  اجرا از طریق toolhub انجام شود. در این حالت reg.run روی آن مهارت خطای
  NotImplementedError می‌دهد (انتظار می‌رود لایه‌ی app از toolhub استفاده کند).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
import importlib
import json

# YAML اختیاری
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# ───────────────────────────── داده/تعریف ─────────────────────────────

@dataclass
class SkillSpec:
    name: str
    kind: str = "skill"                 # "skill" | "tool"
    desc: str = ""
    entry: Optional[str] = None         # "module.sub:func"
    tags: List[str] = field(default_factory=list)
    allowed_args: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    enabled: bool = True

    # بایند اجرا (در فایل ذخیره نمی‌شود)
    _fn: Optional[Callable[..., Dict[str, Any]]] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_fn", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillSpec":
        data = dict(d or {})
        data["name"] = str(data.get("name"))
        data["kind"] = str(data.get("kind", "skill"))
        data["desc"] = str(data.get("desc", ""))
        entry = data.get("entry", None)
        data["entry"] = None if entry in (None, "", "null") else str(entry)
        data["tags"] = list(data.get("tags", []) or [])
        data["allowed_args"] = dict(data.get("allowed_args", {}) or {})
        data["cost"] = float(data.get("cost", 0.0) or 0.0)
        data["enabled"] = bool(data.get("enabled", True))
        return cls(**data)

# ───────────────────────────── رجیستری ─────────────────────────────

class SkillRegistry:
    def __init__(self):
        self._specs: Dict[str, SkillSpec] = {}

    # ثبت/حذف
    def register(self, spec: SkillSpec, fn: Optional[Callable[..., Dict[str, Any]]] = None) -> None:
        if fn is not None:
            spec._fn = fn
        self._specs[spec.name] = spec

    def remove(self, name: str) -> None:
        self._specs.pop(name, None)

    # دسترسی
    def has(self, name: str) -> bool:
        return name in self._specs and self._specs[name].enabled

    def get(self, name: str) -> Optional[SkillSpec]:
        s = self._specs.get(name)
        return s if (s and s.enabled) else None

    def list_all(self) -> List[str]:
        return sorted([n for n, s in self._specs.items() if s.enabled])

    # بایند
    def bind(self, name: str, fn: Callable[..., Dict[str, Any]]) -> None:
        if name not in self._specs:
            self._specs[name] = SkillSpec(name=name, desc="(auto-registered)")
        self._specs[name]._fn = fn

    # بارگذاری از manifest
    def load_manifest(self, path: str | Path = "skills/manifest.yaml") -> int:
        p = Path(path)
        if not p.exists():
            return 0
        text = p.read_text(encoding="utf-8")
        data = {}
        if _HAS_YAML:
            try:
                data = yaml.safe_load(text) or {}
            except Exception:
                data = {}
        if not data:
            try:
                data = json.loads(text)
            except Exception:
                data = {}
        n = 0
        for obj in (data.get("skills") or []):
            try:
                spec = SkillSpec.from_dict(obj or {})
                self._specs[spec.name] = spec
                n += 1
            except Exception:
                continue
        return n

    # Resolve entry → تابع
    @staticmethod
    def _resolve_entry(entry: str) -> Callable[..., Dict[str, Any]]:
        """
        entry شبیه "pkg.module:func" را به تابع پایتون تبدیل می‌کند.
        """
        mod_name, _, func_name = entry.partition(":")
        if not mod_name or not func_name:
            raise ValueError(f"invalid entry spec: {entry!r}")
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name)
        if not callable(fn):
            raise TypeError(f"entry is not callable: {entry!r}")
        return fn

    # اجرا
    def run(self, name: str, /, *args, **kwargs) -> Dict[str, Any]:
        """
        مهارت name را اجرا می‌کند و dict خروجی را برمی‌گرداند.
        اگر entry/bind موجود نباشد → NotImplementedError.
        """
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"skill not found or disabled: {name}")

        fn = spec._fn
        if fn is None and spec.entry:
            fn = self._resolve_entry(spec.entry)
            # cache بعد از resolve
            spec._fn = fn

        if fn is None:
            raise NotImplementedError(f"skill '{name}' has no bound function or entry")

        return fn(*args, **kwargs)

# ───────────────────────────── سازنده‌های سطح‌بالا ─────────────────────────────

def load_skills(path: str | Path = "skills/manifest.yaml") -> SkillRegistry:
    reg = SkillRegistry()
    reg.load_manifest(path)
    return reg

def run_skill(registry: SkillRegistry, name: str, /, *args, **kwargs) -> Dict[str, Any]:
    return registry.run(name, *args, **kwargs)

# ───────────────────────────── تست سریع ─────────────────────────────

if __name__ == "__main__":
    reg = load_skills()
    # اگر reply_greeting در manifest تعریف شده باشد و entry داشته باشد، اجرا می‌شود
    if reg.has("reply_greeting"):
        out = reg.run("reply_greeting", user_text="سلام", plan={"intent":"greeting"})
        print(out.get("text_out"))
    else:
        print("no reply_greeting in manifest")
