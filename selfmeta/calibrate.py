# -*- coding: utf-8 -*-
"""
NOEMA • selfmeta/calibrate.py — کالیبراسیونِ احتمال (V0، فقط stdlib)

هدف:
  - نگاشت «اعتماد خام» p_raw∈[0,1] به احتمالِ کالیبره‌شده p_cal∈[0,1].
  - الگوریتم سبک و آنلاین: Histogram/Bin Binning با هموارسازی لاپلاس و
    میان‌یابی خطی بین بن‌های مجاور. (بدون وابستگی خارجی)

سناریوی استفاده:
    from selfmeta.calibrate import OnlineBinnedCalibrator

    cal = OnlineBinnedCalibrator(bins=12, alpha=1.0, beta=1.0)
    # هر بار که خروجی تولید شد:
    p_raw = 0.78
    outcome_ok = True  # یا False (بر اساس آزمون/بازخورد مربی)
    cal.update(p_raw, 1 if outcome_ok else 0)
    p_cal = cal.predict(p_raw)

    # ذخیره/بارگذاری (فایل JSON سبک)
    cal.save("data/calibration.json")
    cal2 = OnlineBinnedCalibrator.load("data/calibration.json")

ویژگی‌ها:
  - update(p, y): به‌صورت افزایشی شمارنده‌ی بن را به‌روزرسانی می‌کند.
  - predict(p): برای بن مربوطه (و همسایه‌ها) نرخ موفقیت را با هموارسازی
    (Laplace smoothing) برمی‌گرداند و بین دو بن خطی میان‌یابی می‌کند.
  - decay اختیاری: اگر مقدار 0<decay<1 بدهید، حافظه‌ی نمایی‌کاهنده اعمال می‌شود.

یادداشت:
  - این کالیبراتور برای V0 کافی است. برای V1 می‌توانید Isotonic/Platt را
    (با scikit-learn) اضافه کنید و از همین API استفاده نمایید.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json
import math

# ───────────────────────────── ابزار کمکی ─────────────────────────────

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

# ───────────────────────────── مدل ─────────────────────────────

@dataclass
class _Bin:
    n: float = 0.0   # شمار نمونه‌ها (با احتمال، می‌تواند اعشاری باشد اگر decay اعمال شود)
    k: float = 0.0   # شمار موفقیت‌ها (label=1)

class OnlineBinnedCalibrator:
    """
    کالیبراتور آنلاین مبتنی بر بن‌های یکنواخت روی [0,1].
    - bins: تعداد بن‌ها (≥ 2)
    - alpha,beta: پارامترهای هموارسازی لاپلاس (پیشین بتا)
    - min_bin_count: حداقل شمار برای اعتماد به یک بن (کمتر از آن → ترکیب با همسایه)
    - decay: اگر None نباشد و 0<decay<1 → EMA روی شمارنده‌ها اعمال می‌شود.
    """

    def __init__(
        self,
        bins: int = 12,
        alpha: float = 1.0,
        beta: float = 1.0,
        min_bin_count: float = 3.0,
        decay: Optional[float] = None,
    ):
        assert bins >= 2, "bins must be ≥ 2"
        self.bins = int(bins)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_bin_count = float(min_bin_count)
        self.decay = float(decay) if (decay is not None) else None
        self._hist: List[_Bin] = [_Bin() for _ in range(self.bins)]

    # ---------- نگاشت p→idx و مرزها ----------
    def _bin_index(self, p: float) -> int:
        p = _clip01(p)
        i = int(math.floor(p * self.bins))
        if i == self.bins:  # p=1.0
            i = self.bins - 1
        return i

    def _bin_bounds(self, i: int) -> Tuple[float, float]:
        w = 1.0 / self.bins
        a = i * w
        b = (i + 1) * w
        return a, b

    def _bin_center(self, i: int) -> float:
        a, b = self._bin_bounds(i)
        return 0.5 * (a + b)

    # ---------- به‌روزرسانی ----------
    def update(self, p: float, y: int | bool) -> None:
        """
        p: اعتماد خام (0..1)
        y: برچسب دودویی (0/1) — 1=موفق/صحیح
        """
        p = _clip01(float(p))
        i = self._bin_index(p)

        # decay اختیاری
        if self.decay is not None and 0.0 < self.decay < 1.0:
            for b in self._hist:
                b.n *= self.decay
                b.k *= self.decay

        b = self._hist[i]
        b.n += 1.0
        if bool(y):
            b.k += 1.0

    # ---------- پیش‌بینی ----------
    def _bin_rate(self, i: int) -> float:
        b = self._hist[i]
        # هموارسازی لاپلاس: (k+alpha)/(n+alpha+beta)
        return float((b.k + self.alpha) / (b.n + self.alpha + self.beta))

    def predict(self, p: float) -> float:
        """
        احتمال کالیبره‌شده برای p.
        - اگر بن بسیار کم‌نمونه باشد، با همسایه‌ها ترکیب و میان‌یابی می‌شود.
        - برای کاهش ناپیوستگی، بین مراکز بن‌ها خطی میان‌یابی می‌کنیم.
        """
        p = _clip01(float(p))
        i = self._bin_index(p)
        w = 1.0 / self.bins

        # نرخ بن i و همسایه‌های نزدیک
        r_i = self._bin_rate(i)

        # اگر داده‌ی بن کم است، از همسایه‌ها کمک بگیر
        n_i = self._hist[i].n
        if n_i < self.min_bin_count:
            # وزن‌دهی با همسایه‌ها (L/R اگر موجود)
            tot_w = 1.0
            acc = r_i
            if i > 0:
                r_l = self._bin_rate(i - 1)
                wl = min(1.0, self._hist[i - 1].n / max(1.0, self.min_bin_count))
                acc += wl * r_l
                tot_w += wl
            if i < self.bins - 1:
                r_r = self._bin_rate(i + 1)
                wr = min(1.0, self._hist[i + 1].n / max(1.0, self.min_bin_count))
                acc += wr * r_r
                tot_w += wr
            r_i = acc / max(1.0, tot_w)

        # میان‌یابی خطی بین مرکز بن i و همسایه‌ی سمت راست (اگر موجود)
        # t: موقعیت نسبی p داخل بن
        a, b = self._bin_bounds(i)
        t = 0.0 if b == a else (p - a) / (b - a)

        if i < self.bins - 1:
            r_next = self._bin_rate(i + 1)
            # اگر بن بعدی کم‌نمونه بود، نرخ آن را هم نرم کنیم
            if self._hist[i + 1].n < self.min_bin_count and i + 2 < self.bins:
                r_next = 0.5 * r_next + 0.5 * self._bin_rate(i + 2)
            r = (1.0 - t) * r_i + t * r_next
        else:
            # آخرین بن: به بن قبل میان‌یابی کن
            r_prev = self._bin_rate(i - 1) if i > 0 else r_i
            r = (1.0 - t) * r_prev + t * r_i

        return _clip01(float(r))

    # ---------- خروجی نموداری/تحلیلی ----------
    def reliability_curve(self) -> List[Tuple[float, float, float]]:
        """
        لیست (center, rate, n) برای همه‌ی بن‌ها.
        - center: مرکز بن در [0,1]
        - rate: (k+α)/(n+α+β)
        - n: شمار وزن‌دار نمونه‌های بن
        """
        out: List[Tuple[float, float, float]] = []
        for i in range(self.bins):
            c = self._bin_center(i)
            r = self._bin_rate(i)
            n = self._hist[i].n
            out.append((c, r, n))
        return out

    # ---------- ادغام و I/O ----------
    def merge_from(self, other: "OnlineBinnedCalibrator") -> None:
        assert self.bins == other.bins, "bin mismatch"
        for i in range(self.bins):
            self._hist[i].n += other._hist[i].n
            self._hist[i].k += other._hist[i].k

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bins": self.bins,
            "alpha": self.alpha,
            "beta": self.beta,
            "min_bin_count": self.min_bin_count,
            "decay": self.decay,
            "hist": [{"n": b.n, "k": b.k} for b in self._hist],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OnlineBinnedCalibrator":
        obj = cls(
            bins=int(d.get("bins", 12)),
            alpha=float(d.get("alpha", 1.0)),
            beta=float(d.get("beta", 1.0)),
            min_bin_count=float(d.get("min_bin_count", 3.0)),
            decay=float(d["decay"]) if d.get("decay") is not None else None,
        )
        hist = d.get("hist") or []
        for i, bk in enumerate(hist[:obj.bins]):
            obj._hist[i].n = float(bk.get("n", 0.0))
            obj._hist[i].k = float(bk.get("k", 0.0))
        return obj

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "OnlineBinnedCalibrator":
        p = Path(path)
        if not p.exists():
            # اگر فایل نبود، پیش‌فرض برگردان
            return cls()
        obj = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(obj)

# ───────────────────────────── تست سریع ─────────────────────────────

if __name__ == "__main__":
    import random

    cal = OnlineBinnedCalibrator(bins=10, alpha=1.0, beta=1.0, min_bin_count=5.0, decay=0.995)

    # داده‌ی ساختگی: مدل خام کمی خوش‌بین است (over-confident)
    # y ~ Bernoulli( max(0, min(1, p_raw - 0.1)) )
    for _ in range(5000):
        p = random.random()
        true_p = max(0.0, min(1.0, p - 0.10))
        y = 1 if random.random() < true_p else 0
        cal.update(p, y)

    # چند پیش‌بینی نمونه
    for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        print(f"p_raw={q:.2f} → p_cal={cal.predict(q):.3f}")

    # ذخیره و بارگذاری
    cal.save("data/calibration.json")
    cal2 = OnlineBinnedCalibrator.load("data/calibration.json")
    print("bins:", len(cal2.reliability_curve()))
