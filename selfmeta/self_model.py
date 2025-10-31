# -*- coding: utf-8 -*-
"""
NOEMA • selfmeta/self_model.py — مدلِ خود (Self-Model) مینیمال V0
- هدف: نگهداری و به‌روزکردن «وضعیت درونی» نوما به‌صورت یک بردار کم‌بعد.
- خروجی این ماژول برای:
    • لاگ اپیزودیک (memory/episodic.py)
    • تصمیم‌گیری (confidence/uncertainty)
    • خواب/کالیبراسیون (sleep/*, selfmeta/calibrate.py)
  استفاده می‌شود.

❖ ایده:
  self_vector = [
      energy_level,        # [0..1] بودجه/خستگی محاسباتی
      confidence,          # [0..1] اعتماد به پاسخ‌های اخیر (برحسب u→conf)
      surprise_ema,        # [0..1] میانگین متحرک «شگفتی»
      reward_mean,         # [0..1] میانگین پاداش اخیر (کل/شکل‌داده)
      risk_ema,            # [0..1] میانگین ریسک مشاهده‌شده
      uncertainty_ema,     # [0..1] EMA عدم‌قطعیت پیش‌بینی
      progress_ema,        # [0..1] EMA r_int (پیشرفت یادگیری)
      temp_free,           # [0..1] نشانگر آزاد/انسداد (در آینده برای قفل‌ها)
  ]

API اصلی:
    cfg = SelfModelCfg()
    sm  = SelfModel(cfg)
    snap = sm.update(metrics={ "u":0.2, "r_int":0.3, "r_total":0.6, "risk":0.0, "energy":0.05 })
    vec  = sm.vector()  # لیست 8تایی float
    conf = sm.confidence()  # اسکالر [0..1]

توجه:
- این نسخه فقط stdlib را نیاز دارد.
- کالیبراسیون احتمالات در selfmeta/calibrate.py پیاده‌سازی می‌شود و از این‌جا
  می‌توان hook آن را صدا زد (اختیاری).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import math
import time

# ───────────────────────────── پیکربندی ─────────────────────────────

@dataclass
class SelfModelCfg:
    # پنجره‌ی EMA ها
    alpha_surp: float = 0.9          # EMA شگفتی
    alpha_reward: float = 0.9        # EMA پاداش
    alpha_risk: float = 0.9          # EMA ریسک
    alpha_unc: float = 0.9           # EMA عدم‌قطعیت
    alpha_prog: float = 0.9          # EMA پیشرفت (r_int)

    # نگاشت u→confidence
    conf_slope: float = 1.15         # شیب
    conf_bias: float  = 0.05         # بایاس کوچک مثبت

    # بودجه‌ی انرژی (تقریباً «خستگی» محاسباتی)
    energy_decay: float = 0.98        # در هر گام کمی شارژ می‌شود (به 1.0 نزدیک)
    energy_cost_cap: float = 0.25     # سقف هزینهٔ هر گام که لحاظ می‌کنیم
    energy_floor: float = 0.10        # کف انرژی (هرگز به صفر مطلق نرسد)
    energy_init: float = 1.00         # مقدار آغازین

# ───────────────────────────── مدل ─────────────────────────────

class SelfModel:
    """مدلِ خود: بردار وضعیت درونی + به‌روزرسانی‌های EMA."""

    DIM = 8

    def __init__(self, cfg: Optional[SelfModelCfg] = None):
        self.cfg = cfg or SelfModelCfg()
        # حالت درونی
        self._energy = float(self.cfg.energy_init)   # [0..1]
        self._conf   = 0.5                           # [0..1] برآورد اولیه
        self._surp_e = 0.0                           # surprise EMA
        self._rew_e  = 0.0                           # reward EMA
        self._risk_e = 0.0                           # risk EMA
        self._unc_e  = 1.0                           # uncertainty EMA
        self._prog_e = 0.0                           # r_int EMA
        self._free   = 1.0                           # «آزاد بودن» (رزرو برای آینده)
        self._ts     = time.time()

        # هوک کالیبراتور (اختیاری، از selfmeta/calibrate)
        self._calibrator = None  # شیء با method: update(p, y) و predict(p)

    # ───── کمکی‌ها ─────

    @staticmethod
    def _ema(prev: float, now: float, alpha: float) -> float:
        return float(alpha * prev + (1.0 - alpha) * now)

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def _u_to_conf(self, u: float) -> float:
        """
        نگاشت عدم‌قطعیت به اعتماد: conf ≈ 1 - σ(slope*u - bias)
        σ = سیگموید ملایم. سپس کلیپ به [0..1].
        """
        s = self.cfg.conf_slope * float(max(0.0, min(1.0, u))) - self.cfg.conf_bias
        conf = 1.0 - (1.0 / (1.0 + math.exp(-5.0 * (0.5 - s))))  # سیگموید حول 0.5
        return self._clip01(conf)

    # ───── API به‌روزرسانی ─────

    def update(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        metrics انتظار می‌رود شامل برخی از این کلیدها باشد:
          - u: عدم‌قطعیت پیش‌بینی [0..1]
          - r_int: پاداش درونی/پیشرفت [0..1]
          - r_total: پاداش کل (پس از شکل‌دهی) [-1..1] → نگاشت به [0..1]
          - risk: ریسک عمل [0..1]
          - energy: هزینه‌ی انرژی این گام [0..1]
          - surprise: سیگنال شگفتی [0..1] (اگر نبود، با u تقریبی می‌زنیم)

        خروجی: snapshot dict شامل self_vector و مؤلفه‌ها.
        """
        u = float(metrics.get("u", self._unc_e))
        r_int = float(metrics.get("r_int", 0.0))
        r_tot = float(metrics.get("r_total", 0.0))
        risk = float(metrics.get("risk", 0.0))
        energy_cost = float(metrics.get("energy", 0.0))
        surprise = float(metrics.get("surprise", u))

        # 1) انرژی: شارژ ملایم سپس کم‌کردن هزینه
        self._energy = self._clip01(self._energy * self.cfg.energy_decay)
        self._energy = self._clip01(self._energy - min(energy_cost, self.cfg.energy_cost_cap))
        self._energy = max(self._energy, self.cfg.energy_floor)

        # 2) EMA ها
        self._surp_e = self._ema(self._surp_e, surprise, self.cfg.alpha_surp)
        self._rew_e  = self._ema(self._rew_e, (r_tot + 1.0) * 0.5, self.cfg.alpha_reward)  # نگاشت [-1..1]→[0..1]
        self._risk_e = self._ema(self._risk_e, risk, self.cfg.alpha_risk)
        self._unc_e  = self._ema(self._unc_e, u, self.cfg.alpha_unc)
        self._prog_e = self._ema(self._prog_e, r_int, self.cfg.alpha_prog)

        # 3) اعتماد از روی u (و در آینده از کالیبراتور)
        self._conf = self._u_to_conf(self._unc_e)

        # 4) مهر زمان
        self._ts = time.time()

        snap = self.snapshot()
        return snap

    # ───── دسترسی به وضعیت ─────

    def vector(self) -> List[float]:
        """بردار 8بعدی وضعیت درونی را برمی‌گرداند."""
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
        """
        کالکتوری با امضای زیر متصل کنید:
          calibrator.update(p: float, y: int) -> None   # y∈{0,1}
          calibrator.predict(p: float) -> float         # p کالیبره‌شده
        """
        self._calibrator = calibrator

    def calibrated_confidence(self, raw_p: float, outcome_ok: Optional[bool] = None) -> float:
        """
        اگر کالیبراتور نصب باشد و outcome_ok معلوم باشد، آن را به‌روز می‌کند.
        سپس احتمال کالیبره‌شده را برمی‌گرداند.
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

    # چند گام ساختگی
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
