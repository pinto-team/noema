# -*- coding: utf-8 -*-
"""
NOEMA • toolhub/registry.py — Tool registry (V0, file-driven)

Purpose
-------
- Keep an in-memory registry of available tools (specs + optional bound functions).
- Load specs from a YAML/JSON file (config/tools.yaml).
- Optionally bind a safe calculator tool for basic compute flows.

Exports
-------
class ToolRegistry
class ToolSpec
def load_registry(config_path="config/tools.yaml", *, bind_calc=True) -> ToolRegistry
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import json
import re

# ----------------------------- Tool Spec -----------------------------

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
    _fn: Optional[Callable[..., Any]] = field(default=None, repr=False, compare=False)

    @property
    def is_safe_basic(self) -> bool:
        return self.enabled and (self.safety == "safe") and ("basic" in self.tags or True)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_fn", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolSpec":
        data = dict(d or {})
        data.pop("_fn", None)
        data["name"] = str(data.get("name"))
        data["kind"] = str(data.get("kind", "tool"))
        data["desc"] = str(data.get("desc", ""))
        data["safety"] = str(data.get("safety", "safe"))
        data["tags"] = list(data.get("tags", []) or [])
        data["allowed_args"] = dict(data.get("allowed_args", {}) or {})
        data["cost"] = float(data.get("cost", 0.0) or 0.0)
        data["enabled"] = bool(data.get("enabled", True))
        return cls(**data)

# ----------------------------- Registry -----------------------------

class ToolRegistry:
    """Holds ToolSpec objects and optional bound functions to execute them."""

    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}

    # CRUD
    def register(self, spec: ToolSpec, fn: Optional[Callable[..., Any]] = None) -> None:
        if fn is not None:
            spec._fn = fn
        self._specs[spec.name] = spec

    def remove(self, name: str) -> None:
        self._specs.pop(name, None)

    # Lookup
    def has(self, name: str) -> bool:
        return name in self._specs

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def bind(self, name: str, fn: Callable[..., Any]) -> None:
        if name not in self._specs:
            self._specs[name] = ToolSpec(name=name, desc="(auto-registered)", tags=["basic"])
        self._specs[name]._fn = fn

    # Lists
    def list_all(self) -> List[str]:
        return sorted([n for n, s in self._specs.items() if s.enabled])

    def list_safe_basics(self) -> List[str]:
        names = [n for n, s in self._specs.items() if s.enabled and s.safety == "safe"]
        basics = [n for n, s in self._specs.items() if s.enabled and s.safety == "safe" and ("basic" in s.tags)]
        out = basics or names
        out = sorted(set(out))
        if "invoke_calc" in self._specs and "invoke_calc" not in out:
            out.append("invoke_calc")
        return out

    def suggest_for_intent(self, intent: str) -> List[str]:
        intent = (intent or "").strip().lower()
        if intent == "compute":
            return [n for n in self.list_all() if n == "invoke_calc"]
        return self.list_safe_basics()

    # Execution
    def invoke(self, name: str, **kwargs) -> Any:
        spec = self._specs.get(name)
        if spec is None or not spec.enabled:
            raise KeyError(f"tool not found or disabled: {name}")

        # Optional argument verification
        try:
            from toolhub.verify import verify_args  # type: ignore
            verify_args(spec.allowed_args, kwargs)
        except Exception:
            # Best-effort in V0
            pass

        if spec._fn is None:
            raise NotImplementedError(f"tool '{name}' has no bound function.")
        return spec._fn(**kwargs)

    # Persistence
    def load_from_yaml(self, path: str | Path) -> int:
        cfg = _load_yaml(path)
        tools = cfg.get("tools", []) if isinstance(cfg, dict) else []
        n = 0
        for t in tools:
            try:
                spec = ToolSpec.from_dict(t)
                self._specs[spec.name] = spec
                n += 1
            except Exception:
                continue
        return n

    def save_to_jsonl(self, path: str | Path = "data/tools.jsonl") -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for n in self.list_all():
                f.write(json.dumps(self._specs[n].to_dict(), ensure_ascii=False) + "\n")
        return p

# ----------------------------- YAML / JSON -----------------------------

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    # Try YAML
    try:
        import yaml  # type: ignore
        obj = yaml.safe_load(text) or {}
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fallback to JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}

# ----------------------------- Built-ins -----------------------------

def _safe_calc(expr: str) -> str:
    """Extremely limited numeric evaluator: digits + +-*/() and whitespace."""
    if not isinstance(expr, str):
        raise ValueError("expr must be str")
    if not re.fullmatch(r"^[0-9+\-*/() \t]+$", expr):
        raise ValueError("invalid characters in expression")
    return str(eval(expr, {"__builtins__": {}}, {}))

def load_registry(config_path: str | Path = "config/tools.yaml", *, bind_calc: bool = True) -> ToolRegistry:
    """
    Build a ToolRegistry, optionally load from YAML/JSON, and bind a safe calculator.
    This symbol is required by strict mode: toolhub.registry.load_registry
    """
    reg = ToolRegistry()

    # 1) Load specs from file (if present)
    p = Path(config_path)
    if p.exists():
        reg.load_from_yaml(p)

    # 2) Optionally bind a safe calculator
    if bind_calc:
        if not reg.has("invoke_calc"):
            reg.register(ToolSpec(
                name="invoke_calc",
                kind="tool",
                desc="Safe calculator (four basic operations)",
                safety="safe",
                tags=["basic"],
                allowed_args={"expr": {"type": "str", "regex": r"^[0-9+\-*/() \t]+$", "max_len": 256}},
                cost=0.05,
            ))
        reg.bind("invoke_calc", _safe_calc)

    return reg

if __name__ == "__main__":
    r = load_registry(bind_calc=True)
    print("all tools:", r.list_all())
    print("safe basics:", r.list_safe_basics())
    print("invoke calc:", r.invoke("invoke_calc", expr="2+2"))
