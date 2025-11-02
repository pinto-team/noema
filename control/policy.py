# -*- coding: utf-8 -*-
"""
NOEMA • control/policy.py — action selection policy (V0)

Goal
-----
Score each candidate via world.predict() signals and a transparent reward
combiner, then choose either the argmax or a softmax sample.

Signals per candidate (from world.predict):
    (s1, z1_hat, rhat, risk_hat, u_hat)

Scoring (delegates to the value module):
    base = w_int*rhat + w_ext*r_ext − λ*risk − μ*energy
    shaped = base + conf_bonus*conf − u_penalty*u_hat

API
----
decide(state, candidates, predict_fn, *, r_ext=0.0, energy_costs=None, spec=None,
       method="argmax", temperature=0.6) -> (best_action, ranked_details)

Notes
-----
- Safety checks should run before calling the policy (this file doesn't enforce them).
- score_candidate(...) is public for reuse by the planner.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional
import math
import random

# world types (fallback stubs)
try:
    from world import State, Latent, Action  # type: ignore
except Exception:
    @dataclass
    class State:
        s: List[float]; u: float = 0.0; conf: float = 0.0
    @dataclass
    class Latent:
        z: List[float]
    @dataclass
    class Action:
        kind: str; name: str; args: Dict[str, Any]

# value helpers
try:
    from value import RewardSpec, get_default_spec, combine_rewards, shape_bonus
except Exception:
    @dataclass
    class RewardSpec:
        w_int: float = 0.25; w_ext: float = 0.75
        lambda_risk: float = 0.6; mu_energy: float = 0.15
        conf_bonus: float = 0.05; u_penalty: float = 0.05
        clip_min: float = -1.0;  clip_max: float = +1.0
    def get_default_spec(): return RewardSpec()
    def combine_rewards(r_int, r_ext, risk, energy, spec=None):
        sp = spec or get_default_spec()
        val = sp.w_int*r_int + sp.w_ext*r_ext - sp.lambda_risk*max(0.0,risk) - sp.mu_energy*max(0.0,energy)
        return max(sp.clip_min, min(sp.clip_max, float(val)))
    def shape_bonus(r_total, *, confidence, u_hat, spec=None):
        sp = spec or get_default_spec()
        shaped = r_total + sp.conf_bonus*max(0.0,min(1.0,confidence)) - sp.u_penalty*max(0.0,min(1.0,u_hat))
        return max(sp.clip_min, min(sp.clip_max, float(shaped)))

# ------------------------- default energy table -------------------------

_DEFAULT_ENERGY: Dict[str, float] = {
    "reply_greeting": 0.02,
    "invoke_calc":   0.05,
    "ask_clarify":   0.03,
}

def _energy_of(a: Action, table: Optional[Dict[str, float]]) -> float:
    if table and a.name in table:
        return float(table[a.name])
    return float(_DEFAULT_ENERGY.get(a.name, 0.05 if a.kind == "tool" else 0.03))

# ------------------------- scoring -------------------------

def score_candidate(
    state: State,
    action: Action,
    predict_fn: Callable[[State, Action], Tuple[State, Latent, float, float, float]],
    *,
    r_ext: float = 0.0,
    energy_costs: Optional[Dict[str, float]] = None,
    spec: Optional[RewardSpec] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Return (score, details) for one candidate.
    details include: action, s1_conf, rhat, risk, u_hat, energy, base, shaped
    """
    s1, _z1_hat, rhat, risk, u_hat = predict_fn(state, action)
    conf = float(getattr(s1, "conf", 1.0 - float(u_hat)))
    energy = _energy_of(action, energy_costs)

    base = combine_rewards(
        r_int=float(rhat),
        r_ext=float(r_ext),
        risk=float(risk),
        energy=float(energy),
        spec=spec,
    )
    shaped = shape_bonus(base, confidence=conf, u_hat=float(u_hat), spec=spec)

    details = {
        "action": {"kind": action.kind, "name": action.name, "args": dict(action.args or {})},
        "s1_conf": conf,
        "rhat": float(rhat),
        "risk": float(risk),
        "u_hat": float(u_hat),
        "energy": float(energy),
        "base": float(base),
        "shaped": float(shaped),
    }
    return float(shaped), details

# ------------------------- selection -------------------------

def _softmax(xs: List[float], t: float = 0.6) -> List[float]:
    if not xs:
        return []
    t = max(1e-3, float(t))
    m = max(xs)
    exps = [math.exp((x - m) / t) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def decide(
    state: State,
    candidates: List[Action],
    predict_fn: Callable[[State, Action], Tuple[State, Latent, float, float, float]],
    *,
    r_ext: float = 0.0,
    energy_costs: Optional[Dict[str, float]] = None,
    spec: Optional[RewardSpec] = None,
    method: str = "argmax",             # "argmax" | "softmax"
    temperature: float = 0.6,
) -> Tuple[Action, List[Dict[str, Any]]]:
    """
    Choose the best action from candidates.
    Returns (best_action, ranked_details).
    """
    spec = spec or get_default_spec()
    if not candidates:
        a = Action(kind="policy", name="ask_clarify", args={})
        return a, [{"action":{"kind":a.kind,"name":a.name,"args":{}}, "shaped":0.0, "note":"no candidates"}]

    triples: List[Tuple[float, Dict[str, Any], Action]] = []
    for a in candidates:
        sc, det = score_candidate(state, a, predict_fn, r_ext=r_ext, energy_costs=energy_costs, spec=spec)
        triples.append((sc, det, a))

    triples.sort(key=lambda t: t[0], reverse=True)
    scores = [s for s, _, _ in triples]

    if method == "softmax" and len(triples) > 1:
        probs = _softmax(scores, t=float(temperature))
        idx = random.choices(range(len(triples)), weights=probs, k=1)[0]
        best = triples[idx]
    else:
        best = triples[0]

    best_action = best[2]
    ranked_details = [det for _, det, _ in triples]
    return best_action, ranked_details

if __name__ == "__main__":
    s0 = State(s=[0.1]*8, u=0.2, conf=0.8)

    def fake_predict(s: State, a: Action):
        if a.name == "reply_greeting":
            u = 0.1; risk = 0.0; rhat = 0.6
        elif a.name == "invoke_calc":
            u = 0.2; risk = 0.0; rhat = 0.7
        else:
            u = 0.5; risk = 0.06; rhat = 0.2
        s1 = State(s=s.s, u=u, conf=max(0.0, 1.0-u))
        return s1, Latent(z=s.s), rhat, risk, u

    cands = [
        Action(kind="skill",  name="reply_greeting", args={}),
        Action(kind="tool",   name="invoke_calc",    args={"expr":"2+2"}),
        Action(kind="policy", name="ask_clarify",    args={}),
    ]

    a_star, dbg = decide(s0, cands, fake_predict, r_ext=0.0, method="argmax")
    print("best:", a_star)
    for d in dbg:
        print(" -", d["action"]["name"], "score=", round(d["shaped"],3), "| rhat=", d["rhat"], "u=", d["u_hat"])
