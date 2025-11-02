# -*- coding: utf-8 -*-
"""
NOEMA • selfmeta/self_model.py — Minimal Self-Model (V0, cleaned)

- 8D internal state vector with EMA updates.
- Energy model: soft recharge toward 1.0 then subtract per-step cost (capped).
- Confidence: stable logistic mapping from uncertainty u -> confidence in [0..1].
- No external dependencies (stdlib only).

Public API
----------
SelfModelCfg          : dataclass for tunables
SelfModel             : core self-state tracker
  .update(metrics)    : update EMAs and energy; returns a snapshot dict
  .vector()           : 8D internal state vector
  .confidence()       : latest confidence scalar
  .snapshot()         : structured summary for logs
  .attach_calibrator(calibrator)
  .calibrated_confidence(raw_p, outcome_ok=None)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math
import time


# ------------------------------ Config ------------------------------ #

@dataclass
class SelfModelCfg:
    # EMA windows (higher alpha = slower updates)
    alpha_surp: float = 0.90
    alpha_reward: float = 0.90
    alpha_risk: float = 0.90
    alpha_unc: float = 0.90
    alpha_prog: float = 0.90

    # u → confidence via sigmoid: conf = σ(gain * (pivot - u))
    conf_pivot: float = 0.35
    conf_gain: float = 6.0

    # Energy dynamics
    energy_recharge: float = 0.02   # per-step recharge toward 1.0
    energy_cost_cap: float = 0.25   # cap per-step energy cost
    energy_floor: float = 0.10      # never drop below this
    energy_init: float = 1.00       # initial energy


# ------------------------------ Model ------------------------------- #

class SelfModel:
    """Self state: 8D internal vector + EMA updates."""

    DIM = 8

    def __init__(self, cfg: Optional[SelfModelCfg] = None) -> None:
        self.cfg = cfg or SelfModelCfg()

        # internal state
        self._energy = float(self.cfg.energy_init)  # [0..1]
        self._conf = 0.5                            # [0..1]
        self._surp_e = 0.0
        self._rew_e = 0.0
        self._risk_e = 0.0
        self._unc_e = 1.0
        self._prog_e = 0.0
        self._free = 1.0
        self._ts = time.time()

        # optional online calibrator with update(p,y), predict(p)
        self._calibrator: Any = None

    # --------------------------- helpers --------------------------- #

    @staticmethod
    def _ema(prev: float, now: float, alpha: float) -> float:
        return float(alpha * prev + (1.0 - alpha) * now)

    @staticmethod
    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def _u_to_conf(self, u: float) -> float:
        """
        Stable mapping: conf = σ( gain * (pivot - u) ).
        Small u -> high conf. Output is clipped to [0..1].
        """
        u = self._clip01(float(u))
        x = self.cfg.conf_gain * (self.cfg.conf_pivot - u)
        conf = 1.0 / (1.0 + math.exp(-x))
        return self._clip01(conf)

    # --------------------------- updates --------------------------- #

    def update(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update EMAs and energy given metric dict:

        metrics keys (all optional):
          u, r_int, r_total (∈[-1..1]), risk, energy (cost), surprise

        Returns a snapshot dict.
        """
        u = float(metrics.get("u", self._unc_e))
        r_int = float(metrics.get("r_int", 0.0))
        r_tot = float(metrics.get("r_total", 0.0))
        risk = float(metrics.get("risk", 0.0))
        energy_cost = float(metrics.get("energy", 0.0))
        surprise = float(metrics.get("surprise", u))

        # 1) Energy: recharge toward 1.0 then subtract step cost
        self._energy = self._energy + self.cfg.energy_recharge * (1.0 - self._energy)
        self._energy -= min(energy_cost, self.cfg.energy_cost_cap)
        self._energy = self._clip01(max(self._energy, self.cfg.energy_floor))

        # 2) EMA updates (reward mapped from [-1..1] → [0..1])
        self._surp_e = self._ema(self._surp_e, surprise, self.cfg.alpha_surp)
        self._rew_e = self._ema(self._rew_e, (r_tot + 1.0) * 0.5, self.cfg.alpha_reward)
        self._risk_e = self._ema(self._risk_e, risk, self.cfg.alpha_risk)
        self._unc_e = self._ema(self._unc_e, u, self.cfg.alpha_unc)
        self._prog_e = self._ema(self._prog_e, r_int, self.cfg.alpha_prog)

        # 3) Confidence from uncertainty
        self._conf = self._u_to_conf(self._unc_e)

        # 4) Timestamp
        self._ts = time.time()

        return self.snapshot()

    # ------------------------- accessors --------------------------- #

    def vector(self) -> List[float]:
        """8D vector: [energy, conf, surp, rew, risk, unc, prog, free]."""
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
        """Structured summary for logs/debugging."""
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

    # ---------------------- optional calibrator --------------------- #

    def attach_calibrator(self, calibrator: Any) -> None:
        """Attach a calibrator with update(p,y) and predict(p)."""
        self._calibrator = calibrator

    def calibrated_confidence(self, raw_p: float, outcome_ok: Optional[bool] = None) -> float:
        """
        If a calibrator is attached, optionally update it with outcome_ok,
        then return calibrated probability. Falls back to raw_p if anything fails.
        """
        p = self._clip01(float(raw_p))
        if self._calibrator is not None:
            try:
                if outcome_ok is not None:
                    self._calibrator.update(p, 1 if outcome_ok else 0)
                return self._clip01(float(self._calibrator.predict(p)))
            except Exception:
                return p
        return p


# ----------------------------- smoke test ----------------------------- #

if __name__ == "__main__":
    cfg = SelfModelCfg()
    sm = SelfModel(cfg)
    for t in range(5):
        snap = sm.update(
            {
                "u": 0.2 + 0.1 * (t % 2),
                "r_int": 0.3 if t % 2 == 0 else 0.1,
                "r_total": 0.5,
                "risk": 0.0,
                "energy": 0.05,
            }
        )
        print(
            f"[{t}] conf={snap['confidence']:.2f} energy={snap['energy']:.2f} "
            f"uncEMA={snap['uncertainty_ema']:.2f} progEMA={snap['progress_ema']:.2f}"
        )
