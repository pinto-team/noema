# -*- coding: utf-8 -*-
"""
NOEMA • value/reward.py — سیستم ارزش/پاداش مینیمال (V0)
- هدف: یک لایه‌ی شفاف برای محاسبه‌ی پاداش کلّی از اجزای داخلی/بیرونی و قیود.
- با app/main.py سازگار است (در صورت import می‌تواند جایگزین محاسبه‌ی ساده شود).

مفاهیم کلیدی:
  • r_ext : پاداش بیرونی از مربی/محیط (−1..+1)
  • r_int : پاداش درونی (پیشرفت یادگیری/کنجکاوی مفید)
  • risk  : برآورد ریسک عمل (0..1) — جریمه
  • energy: هزینه‌ی محاسبات/منابع (0..1) — جریمه

API:
  spec = get_default_spec()
  r_int, ema = intrinsic_from_errors(prev_ema, err_now, alpha=0.9)
  r_total = combine_rewards(r_int, r_ext, risk, energy, spec)
  shaped = shape_bonus(r_total, confidence=conf, u_hat=u, spec=spec)

نکته:
- این ماژول «تصمیم‌گیر» نیست؛ فقط سیگنال ارزش را تمیز و قابل‌تنظیم می‌کند.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import math

# ----------------------------- پیکربندی -----------------------------

@dataclass
class RewardSpec:
    # وزن‌دهی پایه
    w_int: float = 0.25
    w_ext: float = 0.75
    # جریمه‌ها
    lambda_risk: float = 0.6     # وزن جریمه‌ی ریسک
    mu_energy: float  = 0.15     # وزن جریمه‌ی انرژی/محاسبه
    # شکل‌دهی (اختیاری)
    conf_bonus: float = 0.05     # پاداش کوچک به پاسخ‌های با اعتمادبه‌نفس واقعی
    u_penalty:  float = 0.05     # جریمه‌ی کوچک برای عدم‌قطعیت بالا
    # کلیپ نهایی
    clip_min: float = -1.0
    clip_max: float = +1.0

def get_default_spec() -> RewardSpec:
    return RewardSpec()

# ----------------------------- پاداش درونی: پیشرفت یادگیری -----------------------------

def intrinsic_from_errors(prev_ema_err: float, err_now: float, alpha: float = 0.9) -> Tuple[float, float]:
    """
    r_int = max(0, EMA_prev - EMA_now)
    - prev_ema_err: EMA خطا در گام قبل (هرچه کمتر بهتر)
    - err_now: خطای فعلی مدل (مثلاً MSE پیش‌بینی در نهان‌فضا)
    - alpha: وزن EMA (0.9 یعنی هموارسازی قوی)
    خروجی: (r_int, new_ema)
    """
    ema_now = alpha * prev_ema_err + (1.0 - alpha) * float(err_now)
    r_int = max(0.0, prev_ema_err - ema_now)  # فقط پیشرفت مثبت
    return r_int, ema_now

def intrinsic_from_features(phi_real: float, phi_pred: float) -> float:
    """
    نسخه‌ی بسیار ساده از «کنجکاوی مبتنی بر پیش‌بینی ویژگی»: |Δ|
    در سناریوی واقعی، به جای اسکالر، روی بردار ویژگی MSE/Huber می‌گیرید.
    """
    return abs(float(phi_real) - float(phi_pred))

# ----------------------------- ترکیب اجزا -----------------------------

def combine_rewards(
    r_int: float,
    r_ext: float,
    risk: float,
    energy: float,
    spec: RewardSpec | None = None,
) -> float:
    """
    r_total = w_int * r_int + w_ext * r_ext − λ*risk − μ*energy
    سپس کلیپ به [clip_min, clip_max]
    """
    sp = spec or get_default_spec()
    val = (sp.w_int * float(r_int)) + (sp.w_ext * float(r_ext))
    val -= sp.lambda_risk * max(0.0, float(risk))
    val -= sp.mu_energy  * max(0.0, float(energy))
    return float(max(sp.clip_min, min(sp.clip_max, val)))

def shape_bonus(
    r_total: float,
    *,
    confidence: float,   # 0..1
    u_hat: float,        # 0..1
    spec: RewardSpec | None = None,
) -> float:
    """
    شکل‌دهی ملایم: پاداش کوچک برای پاسخ‌های با اعتمادبه‌نفس کالیبره و جریمه‌ی
    کوچک برای عدم‌قطعیت بالاتر. این کار باعث تثبیت رفتار می‌شود.
    """
    sp = spec or get_default_spec()
    shaped = r_total + sp.conf_bonus * float(max(0.0, min(1.0, confidence)))
    shaped -= sp.u_penalty * float(max(0.0, min(1.0, u_hat)))
    return float(max(sp.clip_min, min(sp.clip_max, shaped)))

# ----------------------------- ابزارهای ایمنی/مقیاس -----------------------------

def safe_combine_dict(signals: Dict[str, float], spec: RewardSpec | None = None) -> float:
    """
    ورودی دیکشنری مانند:
      {"r_int":0.1, "r_ext":1.0, "risk":0.0, "energy":0.05, "conf":0.8, "u_hat":0.2}
    خروجی: r_shaped
    """
    sp = spec or get_default_spec()
    r_int = float(signals.get("r_int", 0.0))
    r_ext = float(signals.get("r_ext", 0.0))
    risk  = float(signals.get("risk", 0.0))
    energy= float(signals.get("energy", 0.0))
    conf  = float(signals.get("conf", 0.0))
    u_hat = float(signals.get("u_hat", 0.0))

    base = combine_rewards(r_int, r_ext, risk, energy, sp)
    shaped = shape_bonus(base, confidence=conf, u_hat=u_hat, spec=sp)
    return shaped

# ----------------------------- تست سریع -----------------------------

if __name__ == "__main__":
    spec = get_default_spec()

    # سناریو ۱: پاداش بیرونی مثبت، ریسک و انرژی کم
    r = combine_rewards(r_int=0.10, r_ext=1.00, risk=0.0, energy=0.05, spec=spec)
    r2 = shape_bonus(r, confidence=0.85, u_hat=0.15, spec=spec)
    print("case1:", r, "→ shaped:", r2)

    # سناریو ۲: بدون پاداش بیرونی، اما پیشرفت یادگیری خوب (r_int بالا)
    r = combine_rewards(r_int=0.35, r_ext=0.0, risk=0.0, energy=0.02, spec=spec)
    r2 = shape_bonus(r, confidence=0.70, u_hat=0.25, spec=spec)
    print("case2:", r, "→ shaped:", r2)

    # سناریو ۳: ریسک بالا → جریمه
    r = combine_rewards(r_int=0.10, r_ext=0.5, risk=0.5, energy=0.1, spec=spec)
    r2 = shape_bonus(r, confidence=0.60, u_hat=0.55, spec=spec)
    print("case3:", r, "→ shaped:", r2)

    # پیشرفت درونی از EMA خطا
    r_int, ema = intrinsic_from_errors(prev_ema_err=0.45, err_now=0.40, alpha=0.9)
    print("intrinsic progress:", r_int, "new_ema:", ema)
