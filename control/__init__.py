# -*- coding: utf-8 -*-
"""
NOEMA • control/init.py — لایه‌ی راه‌انداز کنترل (V0)
- هدف: یک نقطه‌ی واحد برای استفاده از سیاست، برنامه‌ریز و مولد نامزدها.
- این ماژول پیکربندی را می‌خواند (اختیاری)، و API ساده decide() را صادر می‌کند.

وابستگی‌ها:
  - فقط stdlib؛ اگر فایل YAML پیکربندی داشتید، (اختیاری) pyyaml خوانده می‌شود.

استفاده‌ی نمونه:
    from control.init import Control, load_control
    ctl = load_control("config/value.yaml")   # اختیاری؛ اگر نبود، پیش‌فرض می‌گیرد
    a_star, why = ctl.decide(state, plan, r_ext=0.0, predict_fn=world.predict, wm=wm)

قراردادها:
  - state: world.State
  - plan: dict مانند {"intent":"compute","args":{"expr":"2+2"}}
  - predict_fn: همان world.predict
  - r_ext: پاداش بیرونیِ مربی در این گام (−1..+1)
  - wm: حافظه‌ی کاری (اختیاری) برای candidates.generate
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List, Callable
from pathlib import Path
import json

# ماژول‌های کنترل داخلی
from .policy import decide as decide_once, get_default_spec, RewardSpec
from .planner import plan_and_decide
from .candidates import generate as generate_candidates

# انواع world برای تایپ‌هینت
try:
    from world import State, Action, Latent  # type: ignore
except Exception:
    from dataclasses import dataclass
    @dataclass
    class State:
        s: List[float]; u: float = 0.0; conf: float = 0.0
    @dataclass
    class Action:
        kind: str; name: str; args: Dict[str, Any]
    @dataclass
    class Latent:
        z: List[float]

# YAML اختیاری
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# ------------------------------ Config ------------------------------

@dataclass
class ControlCfg:
    # سیاست تک‌گام (اگر planner خاموش باشد)
    method: str = "argmax"          # "argmax" | "softmax"
    temperature: float = 0.6        # برای softmax

    # برنامه‌ریز کوتاه‌مدت
    use_planner: bool = True
    beam: int = 4
    depth: int = 2
    gamma: float = 0.9

    # هزینه‌ی انرژی (اختیاری)
    energy_costs: Optional[Dict[str, float]] = None

    # وزن‌دهی ارزش/پاداش (اگر در value.yaml نبود)
    reward_spec: Optional[Dict[str, float]] = None  # با RewardSpec نگاشت می‌شود

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    if _HAS_YAML:
        try:
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    # fallback: JSON-شکل
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_control(config_path: Optional[str | Path] = None, *, default_energy: Optional[Dict[str, float]] = None) -> "Control":
    """
    تلاش می‌کند از فایل‌های config/* بارگذاری کند:
      - config/value.yaml → reward_spec
      - config/meta.yaml  → روش تصمیم
      - (پارامتر) config_path اگر داده شود، از همان استفاده می‌کند.
    """
    cfg = ControlCfg()
    cfg.energy_costs = default_energy or {}

    # value.yaml
    val_cfg = _load_yaml(config_path or "config/value.yaml")
    if val_cfg:
        # نگاشت کلیدهای آشنا؛ اگر نبودند، نادیده
        rs = {}
        for k in ("w_int","w_ext","lambda_risk","mu_energy","conf_bonus","u_penalty","clip_min","clip_max"):
            if k in val_cfg:
                rs[k] = val_cfg[k]
        cfg.reward_spec = rs or None

    # meta.yaml (اختیاری)
    meta_cfg = _load_yaml("config/meta.yaml")
    if meta_cfg:
        cfg.method = str(meta_cfg.get("control_method", cfg.method))
        cfg.temperature = float(meta_cfg.get("control_temperature", cfg.temperature))
        cfg.use_planner = bool(meta_cfg.get("use_planner", cfg.use_planner))
        cfg.beam = int(meta_cfg.get("planner_beam", cfg.beam))
        cfg.depth = int(meta_cfg.get("planner_depth", cfg.depth))
        cfg.gamma = float(meta_cfg.get("planner_gamma", cfg.gamma))

    return Control(cfg)

# ------------------------------ Core ------------------------------

class Control:
    """
    لایه‌ی یکپارچه‌ی تصمیم:
      - candidates.generate → فهرست اعمال
      - (اختیاری) planner.plan_and_decide → انتخاب دنباله
      - در غیر این صورت policy.decide → انتخاب تک‌گام
    """

    def __init__(self, cfg: Optional[ControlCfg] = None):
        self.cfg = cfg or ControlCfg()
        self.spec = self._make_reward_spec(self.cfg.reward_spec)

    @staticmethod
    def _make_reward_spec(spec_dict: Optional[Dict[str, float]]) -> RewardSpec:
        base = get_default_spec()
        if not spec_dict:
            return base
        # کپی با override
        data = asdict(base)
        data.update({k: float(v) for k, v in spec_dict.items() if k in data})
        return RewardSpec(**data)

    def decide(
        self,
        state: State,
        plan: Dict[str, Any],
        *,
        r_ext: float = 0.0,
        predict_fn: Callable[[State, Action], Tuple[State, Latent, float, float, float]],
        wm: Optional[Any] = None,
    ) -> Tuple[Action, Dict[str, Any]]:
        """
        تصمیمِ نهایی برای یک گام:
          - ورودی: state, plan, r_ext, predict_fn, wm
          - خروجی: (Action, rationale/details)
        """
        # نامزدها
        candidates = generate_candidates(state, plan, wm=wm)

        if self.cfg.use_planner:
            a, rationale = plan_and_decide(
                state, plan, lambda s, p: generate_candidates(s, p, wm=wm),
                predict_fn,
                r_ext=r_ext,
                energy_costs=self.cfg.energy_costs,
                spec=self.spec,
                beam=self.cfg.beam,
                depth=self.cfg.depth,
                gamma=self.cfg.gamma,
                method=self.cfg.method,
            )
            return a, {"by": "planner", **rationale}

        # سیاست تک‌گام
        a, ranked = decide_once(
            state, candidates, predict_fn,
            r_ext=r_ext,
            energy_costs=self.cfg.energy_costs,
            spec=self.spec,
            method=self.cfg.method,
            temperature=self.cfg.temperature,
        )
        return a, {"by": "policy", "ranked": ranked, "intent": plan.get("intent","unknown")}

# ------------------------------ اجرا مستقیم ------------------------------

if __name__ == "__main__":
    # تست سریع با پیش‌بینی ساختگی
    s0 = State(s=[0.1]*8, u=0.2, conf=0.8)

    def fake_predict(s: State, a: Action):
        if a.name == "reply_greeting":
            u = 0.1; risk = 0.0; rhat = 0.6
        elif a.name == "invoke_calc":
            u = 0.2; risk = 0.0; rhat = 0.7
        elif a.name == "ask_clarify":
            u = 0.05; risk = 0.0; rhat = 0.4
        else:
            u = 0.5; risk = 0.06; rhat = 0.2
        s1 = State(s=s.s, u=u, conf=max(0.0, 1.0-u))
        return s1, Latent(z=s.s), rhat, risk, u

    ctl = load_control()
    # 1) greet
    a1, why1 = ctl.decide(s0, {"intent":"greeting"}, r_ext=0.0, predict_fn=fake_predict)
    print("GREETING:", a1, "|", why1)
    # 2) compute
    a2, why2 = ctl.decide(s0, {"intent":"compute","args":{"expr":"2+2"}}, r_ext=0.0, predict_fn=fake_predict)
    print("COMPUTE :", a2, "|", why2)
    # 3) unknown
    a3, why3 = ctl.decide(s0, {"intent":"unknown"}, r_ext=0.0, predict_fn=fake_predict)
    print("UNKNOWN :", a3, "|", why3)
