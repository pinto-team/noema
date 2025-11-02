# -*- coding: utf-8 -*-
"""
NOEMA • selfmeta/self_model.py — مدلِ خود (Self-Model) مینیمال V0 (clean)
- بردار وضعیت درونی 8بُعدی + به‌روزرسانی‌های EMA.
- اصلاحات:
  • انرژی: شارژ ملایم به‌سمت 1.0 سپس کسر هزینهٔ گام (به‌جای decay کاهنده).
  • u→confidence: لاجیت پایدار با pivot/gain قابل‌تنظیم (برای u کم، conf زیاد).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math
import time

# ───────────────────────────── پیکربندی ─────────────────────────────

@dataclass
class SelfModelCfg:
    # EMA windows
    alpha_surp: float = 0.9
    alpha_reward: float = 0.9
    alpha_risk: float = 0.9
    alpha_unc: float = 0.9
    alpha_prog: float = 0.9

    # u → confidence (sigmoid on (pivot - u))
    conf_pivot: float = 0.35      # نقطهٔ تعادل u که اطرافش conf≈0.5
    conf_gain: float  = 6.0       # تیزی منحنی (بزرگ‌تر = اشباع سریع‌تر)

    # انرژی (شارژ به‌سمت 1.0)
    energy_recharge: float = 0.02 # نرخ شارژ در هر گام (۰..۱)
    energy_cost_cap: float = 0.25 # سقف هزینهٔ لحاظ‌شده برای هر گام
    energy_floor: float = 0.10    # کف انرژی
    energy_init: float = 1.00     # مقدار آغازین

# ───────────────────────────── مدل ─────────────────────────────

class SelfModel:
    """مدلِ خود: بردار وضعیت درونی + به‌روزرسانی‌های EMA."""

    DIM = 8

    def __init__(self, cfg: Optional[SelfModelCfg] = None):
        self.cfg = cfg or SelfModelCfg()
        # حالت درونی
        self._energy = float(self.cfg.energy_init)   # [0..1]
        self._conf   = 0.5                           # [0..1]
        self._surp_e = 0.0
        self._rew_e  = 0.0
        self._risk_e = 0.0
        self._unc_e  = 1.0
        self._prog_e = 0.0
        self._free   = 1.0
        self._ts     = time.time()

        # هوک کالیبراتور (اختیاری)
        self._calibrator = None  # شیء با متدهای update(p,y) و predict(p)

    # ───── کمکی‌ها ─────

    @staticmethod
    def _ema(prev: float, now: float, alpha: float) -> float:
        return float(alpha * prev + (1.0 - alpha) * now)

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def _u_to_conf(self, u: float) -> float:
        """
        نگاشت پایدار: conf = σ( gain * (pivot - u) )
        - u کوچک‌تر از pivot → conf بزرگ‌تر (رفتار شهودی)
        """
        u = self._clip01(float(u))
        x = self.cfg.conf_gain * (self.cfg.conf_pivot - u)
        conf = 1.0 / (1.0 + math.exp(-x))
        return self._clip01(conf)

    # ───── API به‌روزرسانی ─────

    def update(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        metrics می‌تواند شامل کلیدهای زیر باشد:
          u, r_int, r_total ([-1..1]), risk, energy, surprise
        خروجی: snapshot شامل مؤلفه‌ها و self_vector.
        """
        u = float(metrics.get("u", self._unc_e))
        r_int = float(metrics.get("r_int", 0.0))
        r_tot = float(metrics.get("r_total", 0.0))
        risk = float(metrics.get("risk", 0.0))
        energy_cost = float(metrics.get("energy", 0.0))
        surprise = float(metrics.get("surprise", u))

        # 1) انرژی: شارژ به‌سمت 1.0 سپس کسر هزینه
        # energy ← energy + η*(1-energy) − cost
        self._energy = self._energy + self.cfg.energy_recharge * (1.0 - self._energy)
        self._energy = self._energy - min(energy_cost, self.cfg.energy_cost_cap)
        self._energy = self._clip01(max(self._energy, self.cfg.energy_floor))

        # 2) EMA ها
        self._surp_e = self._ema(self._surp_e, surprise, self.cfg.alpha_surp)
        self._rew_e  = self._ema(self._rew_e, (r_tot + 1.0) * 0.5, self.cfg.alpha_reward)  # [-1..1]→[0..1]
        self._risk_e = self._ema(self._risk_e, risk, self.cfg.alpha_risk)
        self._unc_e  = self._ema(self._unc_e, u, self.cfg.alpha_unc)
        self._prog_e = self._ema(self._prog_e, r_int, self.cfg.alpha_prog)

        # 3) اعتماد از روی u (کالیبره‌کردنِ اختیاری بیرون از این تابع انجام می‌شود)
        self._conf = self._u_to_conf(self._unc_e)

        # 4) مهر زمان
        self._ts = time.time()

        return self.snapshot()

    # ───── دسترسی به وضعیت ─────

    def vector(self) -> List[float]:
        """بردار 8بعدی وضعیت درونی را برمی‌گرداند: [energy, conf, surp, rew, risk, unc, prog, free]."""
        return [
            self._energy,
            self._conf,
            self._surp_e,
            self._rew_e,
            self._risk_e,
            self._unc_e,
            self._prog_e,
            self._free,
        ]

    def confidence(self) -> float:
        return float(self._conf)

    def snapshot(self) -> Dict[str, Any]:
        """خلاصه‌ی وضعیت برای لاگ/دیباگ."""
        return {
            "ts": self._ts,
            "energy": self._energy,
            "confidence": self._conf,
            "surprise_ema": self._surp_e,
            "reward_mean": self._rew_e,
            "risk_ema": self._risk_e,
            "uncertainty_ema": self._unc_e,
            "progress_ema": self._prog_e,
            "vector": self.vector(),
        }

    # ───── کالیبراسیون (اختیاری) ─────

    def attach_calibrator(self, calibrator: Any) -> None:
        """calibrator با امضای update(p,y) و predict(p) را متصل کنید."""
        self._calibrator = calibrator

    def calibrated_confidence(self, raw_p: float, outcome_ok: Optional[bool] = None) -> float:
        """
        اگر کالیبراتور نصب باشد و outcome_ok معلوم باشد، آن را به‌روز می‌کند،
        سپس p کالیبره‌شده را برمی‌گرداند.
        """
        p = self._clip01(float(raw_p))
        if self._calibrator is not None:
            if outcome_ok is not None:
                try:
                    self._calibrator.update(p, 1 if outcome_ok else 0)
                except Exception:
                    pass
            try:
                return self._clip01(float(self._calibrator.predict(p)))
            except Exception:
                return p
        return p

# ───────────────────────────── تست سریع ─────────────────────────────

if __name__ == "__main__":
    cfg = SelfModelCfg()
    sm = SelfModel(cfg)
    for t in range(5):
        snap = sm.update({
            "u": 0.2 + 0.1 * (t % 2),
            "r_int": 0.3 if t % 2 == 0 else 0.1,
            "r_total": 0.5,
            "risk": 0.0,
            "energy": 0.05,
        })
        print(f"[{t}] conf={snap['confidence']:.2f} energy={snap['energy']:.2f} "
              f"uncEMA={snap['uncertainty_ema']:.2f} progEMA={snap['progress_ema']:.2f}")
