# -*- coding: utf-8 -*-
"""
NOEMA • selfmeta/calibrate.py — Online probability calibration via binned stats (V0)

- Maps raw confidence p_raw ∈ [0,1] to calibrated probability p_cal ∈ [0,1].
- Lightweight, online, stdlib-only: histogram/binning with Laplace smoothing.
- Optional exponential decay for forgetting.
- Linear interpolation between adjacent bins for smoothness.

Public API
----------
OnlineBinnedCalibrator
  .update(p, y)
  .predict(p)
  .reliability_curve()
  .merge_from(other)
  .to_dict() / from_dict()
  .save(path) / load(path)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import math


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ------------------------------ Model ------------------------------ #

@dataclass
class _Bin:
    n: float = 0.0  # weighted count (decay may make it fractional)
    k: float = 0.0  # weighted successes


class OnlineBinnedCalibrator:
    """
    Online calibrator using uniform bins over [0,1].

    Params
    ------
    bins : int         number of bins (>=2)
    alpha, beta : float  Laplace smoothing (Beta prior)
    min_bin_count : float  below this, blend with neighbors
    decay : Optional[float]  if 0<decay<1, apply EMA on counts (forgetting)
    """

    def __init__(
        self,
        bins: int = 12,
        alpha: float = 1.0,
        beta: float = 1.0,
        min_bin_count: float = 3.0,
        decay: Optional[float] = None,
    ) -> None:
        if bins < 2:
            raise ValueError("bins must be >= 2")
        self.bins = int(bins)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_bin_count = float(min_bin_count)
        self.decay = float(decay) if (decay is not None) else None
        self._hist: List[_Bin] = [_Bin() for _ in range(self.bins)]

    # ------------------------- bin mapping ------------------------- #

    def _bin_index(self, p: float) -> int:
        p = _clip01(p)
        i = int(math.floor(p * self.bins))
        if i == self.bins:  # p=1.0 edge
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

    # --------------------------- update ---------------------------- #

    def update(self, p: float, y: int | bool) -> None:
        """Incrementally update the histogram with observation (p, y∈{0,1})."""
        p = _clip01(float(p))

        # optional decay
        if self.decay is not None and 0.0 < self.decay < 1.0:
            for b in self._hist:
                b.n *= self.decay
                b.k *= self.decay

        i = self._bin_index(p)
        b = self._hist[i]
        b.n += 1.0
        if bool(y):
            b.k += 1.0

    # -------------------------- inference -------------------------- #

    def _bin_rate(self, i: int) -> float:
        b = self._hist[i]
        # Laplace smoothing
        return float((b.k + self.alpha) / (b.n + self.alpha + self.beta))

    def predict(self, p: float) -> float:
        """
        Return calibrated probability for raw p.
        Uses neighbor blending when counts are low and linear interpolation
        inside the bin for smoothness.
        """
        p = _clip01(float(p))
        i = self._bin_index(p)

        # base rate for bin i
        r_i = self._bin_rate(i)

        # neighbor blending if under-sampled
        n_i = self._hist[i].n
        if n_i < self.min_bin_count:
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

        # linear interpolation within bin
        a, b = self._bin_bounds(i)
        t = 0.0 if b == a else (p - a) / (b - a)

        if i < self.bins - 1:
            r_next = self._bin_rate(i + 1)
            if self._hist[i + 1].n < self.min_bin_count and i + 2 < self.bins:
                r_next = 0.5 * r_next + 0.5 * self._bin_rate(i + 2)
            r = (1.0 - t) * r_i + t * r_next
        else:
            r_prev = self._bin_rate(i - 1) if i > 0 else r_i
            r = (1.0 - t) * r_prev + t * r_i

        return _clip01(float(r))

    # -------------------------- analytics -------------------------- #

    def reliability_curve(self) -> List[Tuple[float, float, float]]:
        """Return [(center, rate, n_weight)] for all bins."""
        out: List[Tuple[float, float, float]] = []
        for i in range(self.bins):
            c = self._bin_center(i)
            r = self._bin_rate(i)
            n = self._hist[i].n
            out.append((c, r, n))
        return out

    # -------------------------- I/O & merge ------------------------- #

    def merge_from(self, other: "OnlineBinnedCalibrator") -> None:
        if self.bins != other.bins:
            raise ValueError("bin mismatch")
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
        for i, bk in enumerate(hist[: obj.bins]):
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
            return cls()
        obj = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(obj)


# ----------------------------- smoke test ----------------------------- #

if __name__ == "__main__":
    import random

    cal = OnlineBinnedCalibrator(bins=10, alpha=1.0, beta=1.0, min_bin_count=5.0, decay=0.995)

    # synthetic: over-confident model
    for _ in range(5000):
        p = random.random()
        true_p = max(0.0, min(1.0, p - 0.10))
        y = 1 if random.random() < true_p else 0
        cal.update(p, y)

    for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        print(f"p_raw={q:.2f} → p_cal={cal.predict(q):.3f}")

    cal.save("data/calibration.json")
    cal2 = OnlineBinnedCalibrator.load("data/calibration.json")
    print("bins:", len(cal2.reliability_curve()))
