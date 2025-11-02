# -*- coding: utf-8 -*-
"""
NOEMA • world/dynamics.py — minimal, dependency-light world model (V0)

Goal:
  - Build a stable State from the recent latent history (z) and
    predict the next latent given an Action.
  - Provide heuristic estimates for predictive uncertainty (u_hat) and risk.

Design notes:
  - Numpy-only, no heavy ML stack. The API matches what app/main.py expects:
      state(z_hist) -> State
      predict(s, a) -> (State, Latent, r_hat, risk_hat, u_hat)
  - Deterministic action embedding via a stable hash, so tests are reproducible.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import hashlib
import math
import struct

import numpy as np

__all__ = ["Latent", "State", "Action", "state", "predict"]

# ------------------------- Minimal types -------------------------

@dataclass
class Latent:
    z: List[float]

@dataclass
class State:
    s: List[float]
    u: float = 0.0   # uncertainty in [0, 1]
    conf: float = 0.0  # confidence = 1 - u

@dataclass
class Action:
    kind: str                 # "skill" | "tool" | "policy"
    name: str                 # action name
    args: Dict[str, Any]      # action args


# ------------------------- Utilities -------------------------

def _to_np(x: List[float]) -> np.ndarray:
    """Coerce a Python list of floats to float32 numpy array."""
    return np.asarray(list(x or []), dtype=np.float32)

def _from_np(x: np.ndarray) -> List[float]:
    """Coerce a numpy vector to a Python list of float32."""
    return x.astype(np.float32).tolist()

def _stable_float_stream(key: str, n: int) -> np.ndarray:
    """
    Produce a deterministic sequence of n floats in [-1, 1] using a hash chain.
    Avoids any global RNG state; good for tests and reproducibility.
    """
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    out = np.zeros((n,), dtype=np.float32)
    seed = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    # Expand via chained blake2b until we have at least n*4 bytes.
    buf = bytearray(seed)
    while len(buf) < n * 4:
        seed = hashlib.blake2b(seed, digest_size=16).digest()
        buf.extend(seed)
    # Map 4 bytes → uint32 → [0,1) → [-1,1]
    for i in range(n):
        chunk = bytes(buf[4 * i : 4 * i + 4])
        v01 = struct.unpack(">I", chunk)[0] / 0xFFFFFFFF
        out[i] = 2.0 * v01 - 1.0
    # Normalize for a stable magnitude
    norm = float(np.linalg.norm(out)) or 1.0
    return (out / norm).astype(np.float32)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]."""
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / (na * nb))


# ------------------------- State builder -------------------------

# Heuristic scale for mapping mean absolute deviation to [0..1]
_MAD_SCALE = 0.25

def state(z_hist: List[Latent]) -> State:
    """
    Build a stable State from recent latents:
      - s = mean of z vectors
      - u = normalized mean absolute deviation (MAD) mapped into [0, 1]
      - conf = 1 - u
    """
    if not z_hist:
        return State(s=[], u=1.0, conf=0.0)

    # Filter out empty vectors and check dimensional consistency
    zs = [z.z for z in z_hist if isinstance(z, Latent) and z.z]
    if not zs:
        return State(s=[], u=1.0, conf=0.0)

    # Pad or trim to a common dimension if needed (use the first vector as reference)
    D = len(zs[0])
    zs_aligned: List[np.ndarray] = []
    for vec in zs:
        if len(vec) == D:
            zs_aligned.append(_to_np(vec))
        elif len(vec) > D:
            zs_aligned.append(_to_np(vec[:D]))
        else:
            pad = np.zeros((D,), dtype=np.float32)
            v = _to_np(vec)
            pad[: v.shape[0]] = v
            zs_aligned.append(pad)

    Z = np.vstack(zs_aligned)  # [T, D]
    s_vec = Z.mean(axis=0)

    # Uncertainty via normalized MAD
    mad = float(np.mean(np.abs(Z - s_vec[None, :])))
    u = float(np.clip(mad / _MAD_SCALE, 0.0, 1.0))
    conf = float(np.clip(1.0 - u, 0.0, 1.0))

    return State(s=_from_np(s_vec), u=u, conf=conf)


# ------------------------- Dynamics / Prediction -------------------------

# Linear dynamics coefficients: s1 = ALPHA * s + BETA * a_embed
_ALPHA = 0.88  # inertia
_BETA  = 0.12  # action influence

# Soft norm clamp for s1
_S1_NORM_CLIP = 1.8

# Action names considered very low risk (tunable)
_LOW_RISK_ACTIONS = {"reply_greeting", "invoke_calc", "ask_clarify"}

def _action_embed(a: Action, dim: int) -> np.ndarray:
    """
    Deterministic action embedding in R^dim:
      - Base direction from action name via a stable hash stream.
      - Small scalar tweak from args (e.g., expression length) to encode effort/shape.
    """
    base = _stable_float_stream(f"action::{a.name}", dim)
    bonus = 0.0
    if isinstance(a.args, dict) and a.args:
        if isinstance(a.args.get("expr"), str):
            bonus = min(len(a.args["expr"]) / 64.0, 1.0)
        else:
            bonus = min(len(a.args.keys()) / 8.0, 1.0)
    emb = np.clip(base + 0.05 * bonus, -1.0, 1.0)
    return emb.astype(np.float32)

def _uncertainty_hat(s_vec: np.ndarray, s1_vec: np.ndarray, a: Action) -> float:
    """
    Predictive uncertainty increases with relative state change and action novelty.
    Returns a value in [0, 1].
    """
    denom = float(np.linalg.norm(s_vec)) + 1e-6
    delta_rel = float(np.linalg.norm(s1_vec - s_vec)) / denom
    novelty = 0.35 if a.name not in _LOW_RISK_ACTIONS else 0.10
    u_hat = np.clip(0.20 * delta_rel + novelty, 0.0, 1.0)
    return float(u_hat)

def _risk_hat(a: Action) -> float:
    """Heuristic risk: near-zero for whitelisted actions, small otherwise."""
    return 0.0 if a.name in _LOW_RISK_ACTIONS else 0.06

def _rhat_progress(s_vec: np.ndarray, s1_vec: np.ndarray) -> float:
    """
    Expected intrinsic progress proxy in [0,1], using cosine similarity
    between current and predicted next state.
    """
    c = _cosine(s_vec, s1_vec)  # [-1, 1]
    return float(0.5 * (c + 1.0))  # map to [0, 1]

def predict(s: State, a: Action) -> Tuple[State, Latent, float, float, float]:
    """
    Predict next state and latent given current state and an action.

    Returns:
      - s1: next State
      - z1_hat: predicted next Latent
      - r_hat: expected intrinsic progress (0..1)
      - risk_hat: heuristic risk (0..~0.1)
      - u_hat: predictive uncertainty (0..1)
    """
    if not isinstance(s, State) or not s.s:
        # Empty state: return a neutral prediction with max uncertainty
        return State(s=[], u=1.0, conf=0.0), Latent([]), 0.0, 0.0, 1.0

    s_vec = _to_np(s.s)
    dim = int(s_vec.shape[0])
    if dim == 0:
        return State(s=[], u=1.0, conf=0.0), Latent([]), 0.0, 0.0, 1.0

    a_vec = _action_embed(a, dim)

    # Simple linear dynamics with soft norm control
    s1_vec = (_ALPHA * s_vec + _BETA * a_vec).astype(np.float32)
    norm = float(np.linalg.norm(s1_vec))
    if norm > _S1_NORM_CLIP:
        s1_vec = (s1_vec / norm).astype(np.float32)

    # Uncertainty and risk
    u_hat = _uncertainty_hat(s_vec, s1_vec, a)
    risk = _risk_hat(a)

    # Confidence of next state = 1 - u_hat
    conf1 = float(np.clip(1.0 - u_hat, 0.0, 1.0))
    s1 = State(s=_from_np(s1_vec), u=u_hat, conf=conf1)

    # Predicted latent = next state (intra-model latent)
    z1_hat = Latent(_from_np(s1_vec))

    # Expected intrinsic progress
    r_hat = _rhat_progress(s_vec, s1_vec)

    return s1, z1_hat, r_hat, risk, u_hat


# ------------------------- Manual test -------------------------

if __name__ == "__main__":
    # Build a state from a tiny synthetic history
    z_hist = [Latent([0.10] * 64), Latent([0.12] * 64), Latent([0.11] * 64)]
    s0 = state(z_hist)
    print(f"s0: u={s0.u:.3f}, conf={s0.conf:.3f}, ||s||={math.sqrt(sum(x*x for x in s0.s)):.3f}")

    a = Action(kind="tool", name="invoke_calc", args={"expr": "2+2"})
    s1, z1h, rhat, risk, uhat = predict(s0, a)
    print(f"predict: r_hat={rhat:.3f}, risk={risk:.3f}, u_hat={uhat:.3f}, ||z1||={math.sqrt(sum(x*x for x in z1h.z)):.3f}")
