# -*- coding: utf-8 -*-
"""
NOEMA • value/reward.py — Minimal value/reward system (V0)

Purpose:
  A small, transparent layer to compute total reward from intrinsic/extrinsic
  signals plus simple penalties and shaping. Designed to plug into app/main.py.

Concepts:
  • r_ext : external reward from coach/environment (−1..+1)
  • r_int : intrinsic reward (learning progress / curiosity)
  • risk  : estimated risk of the chosen action (0..1) — penalty
  • energy: compute/resource cost (0..1) — penalty

API:
  spec = get_default_spec()
  r_int, ema = intrinsic_from_errors(prev_ema, err_now, alpha=0.9)
  r_total = combine_rewards(r_int, r_ext, risk, energy, spec)
  shaped = shape_bonus(r_total, confidence=conf, u_hat=u, spec=spec)

This module does not make decisions; it only provides value signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


# ----------------------------- Config -----------------------------

@dataclass
class RewardSpec:
    # base weighting
    w_int: float = 0.25
    w_ext: float = 0.75
    # penalties
    lambda_risk: float = 0.6     # risk penalty weight
    mu_energy: float = 0.15      # energy/compute penalty weight
    # mild shaping
    conf_bonus: float = 0.05     # small bonus for calibrated confidence
    u_penalty: float = 0.05      # small penalty for higher uncertainty
    # final clipping
    clip_min: float = -1.0
    clip_max: float = +1.0


def get_default_spec() -> RewardSpec:
    return RewardSpec()


# ----------------------------- Intrinsic reward -----------------------------

def intrinsic_from_errors(
    prev_ema_err: float,
    err_now: float,
    alpha: float = 0.9,
) -> Tuple[float, float]:
    """
    Learning progress via EMA delta:
      ema_now = alpha * prev_ema_err + (1 - alpha) * err_now
      r_int   = max(0, prev_ema_err - ema_now)
    Returns (r_int, ema_now).
    """
    ema_now = alpha * float(prev_ema_err) + (1.0 - alpha) * float(err_now)
    r_int = max(0.0, float(prev_ema_err) - ema_now)
    return r_int, ema_now


def intrinsic_from_features(phi_real: float, phi_pred: float) -> float:
    """
    Extremely simple prediction-error curiosity on scalar features.
    In practice, use vector losses (e.g., MSE/Huber) over embeddings.
    """
    return abs(float(phi_real) - float(phi_pred))


# ----------------------------- Combination -----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def combine_rewards(
    r_int: float,
    r_ext: float,
    risk: float,
    energy: float,
    spec: RewardSpec | None = None,
) -> float:
    """
    Total (pre-shaping):
      r_total = w_int * r_int + w_ext * r_ext − λ * risk − μ * energy
      then clipped to [clip_min, clip_max].
    """
    sp = spec or get_default_spec()
    val = (sp.w_int * float(r_int)) + (sp.w_ext * float(r_ext))
    val -= sp.lambda_risk * max(0.0, float(risk))
    val -= sp.mu_energy * max(0.0, float(energy))
    return _clamp(val, sp.clip_min, sp.clip_max)


def shape_bonus(
    r_total: float,
    *,
    confidence: float,   # 0..1
    u_hat: float,        # 0..1
    spec: RewardSpec | None = None,
) -> float:
    """
    Mild shaping for stability:
      + small bonus for calibrated confidence
      - small penalty for higher uncertainty
    """
    sp = spec or get_default_spec()
    shaped = float(r_total)
    shaped += sp.conf_bonus * _clamp(float(confidence), 0.0, 1.0)
    shaped -= sp.u_penalty * _clamp(float(u_hat), 0.0, 1.0)
    return _clamp(shaped, sp.clip_min, sp.clip_max)


# ----------------------------- Safe dict helper -----------------------------

def safe_combine_dict(signals: Dict[str, float], spec: RewardSpec | None = None) -> float:
    """
    Combine from a dict like:
      {"r_int":0.1, "r_ext":1.0, "risk":0.0, "energy":0.05, "conf":0.8, "u_hat":0.2}
    Returns shaped reward (float).
    """
    sp = spec or get_default_spec()
    r_int = float(signals.get("r_int", 0.0))
    r_ext = float(signals.get("r_ext", 0.0))
    risk = float(signals.get("risk", 0.0))
    energy = float(signals.get("energy", 0.0))
    conf = float(signals.get("conf", 0.0))
    u_hat = float(signals.get("u_hat", 0.0))

    base = combine_rewards(r_int, r_ext, risk, energy, sp)
    shaped = shape_bonus(base, confidence=conf, u_hat=u_hat, spec=sp)
    return shaped


# ----------------------------- Quick self-test -----------------------------

if __name__ == "__main__":
    spec = get_default_spec()

    # Case 1: positive external reward, low risk/energy
    r = combine_rewards(r_int=0.10, r_ext=1.00, risk=0.0, energy=0.05, spec=spec)
    r2 = shape_bonus(r, confidence=0.85, u_hat=0.15, spec=spec)
    print("case1:", r, "→ shaped:", r2)

    # Case 2: no external reward, good learning progress
    r = combine_rewards(r_int=0.35, r_ext=0.0, risk=0.0, energy=0.02, spec=spec)
    r2 = shape_bonus(r, confidence=0.70, u_hat=0.25, spec=spec)
    print("case2:", r, "→ shaped:", r2)

    # Case 3: higher risk → penalty
    r = combine_rewards(r_int=0.10, r_ext=0.5, risk=0.5, energy=0.1, spec=spec)
    r2 = shape_bonus(r, confidence=0.60, u_hat=0.55, spec=spec)
    print("case3:", r, "→ shaped:", r2)

    # Intrinsic from EMA errors
    r_int, ema = intrinsic_from_errors(prev_ema_err=0.45, err_now=0.40, alpha=0.9)
    print("intrinsic progress:", r_int, "new_ema:", ema)
