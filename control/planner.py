# -*- coding: utf-8 -*-
"""
NOEMA • control/planner.py — short-horizon planner (V0, lightweight)

Goal
-----
Given the current State, a high-level plan/intent, a candidate generator,
and the world model's predict() function, simulate short action sequences
(depth 1..2), score them, and return the *first* action from the best
sequence plus a rationale for debugging.

API
----
plan_and_decide(
    state, plan, generate_candidates_fn, predict_fn,
    *, r_ext=0.0, beam=3, depth=2, gamma=0.9, method="argmax"
) -> (best_action, rationale)

Notes
-----
- `generate_candidates_fn` is what you expose from control/candidates.py (or app/main.py).
- `predict_fn` is typically world.predict.
- Scoring per step is delegated to control.policy.score_candidate to stay
  consistent with the policy.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# world types (fallback stubs for type hints)
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

# policy helpers
from .policy import score_candidate, RewardSpec, get_default_spec

# ----------------------------- helpers -----------------------------

def _default_seq_generator(
    state: State,
    plan: Dict[str, Any],
    base_candidates: List[Action],
) -> List[List[Action]]:
    """
    Build simple sequences:
      - length 1: all base candidates
      - length 2:
         * unknown → [ask_clarify → <each base>]
         * compute → [ask_clarify → invoke_calc] when calc exists
    """
    seqs: List[List[Action]] = []

    # len=1
    for a in base_candidates:
        seqs.append([a])

    intent = plan.get("intent", "unknown")
    if intent == "unknown":
        clarify = Action(kind="policy", name="ask_clarify", args={})
        for a in base_candidates:
            if a.name != "ask_clarify":
                seqs.append([clarify, a])
    elif intent == "compute":
        clarify = Action(kind="policy", name="ask_clarify", args={})
        calc = next((a for a in base_candidates if a.name == "invoke_calc"), None)
        if calc:
            seqs.append([clarify, calc])

    # de-duplicate
    uniq: List[List[Action]] = []
    seen = set()
    for seq in seqs:
        key = tuple((a.kind, a.name, tuple(sorted((a.args or {}).items()))) for a in seq)
        if key not in seen:
            uniq.append(seq)
            seen.add(key)
    return uniq

def _rollout_sequence(
    state: State,
    seq: List[Action],
    predict_fn: Callable[[State, Action], Tuple[State, Latent, float, float, float]],
    *,
    r_ext: float = 0.0,
    spec: Optional[RewardSpec] = None,
    energy_costs: Optional[Dict[str, float]] = None,
    gamma: float = 0.9,
) -> Tuple[float, List[Dict[str, Any]], State]:
    """
    Roll out a short sequence with discount gamma and return:
      (total_score, step_details[], final_state)
    """
    spec = spec or get_default_spec()
    total = 0.0
    details: List[Dict[str, Any]] = []
    s = state
    discount = 1.0

    for step_idx, a in enumerate(seq):
        score, det = score_candidate(s, a, predict_fn, r_ext=r_ext, energy_costs=energy_costs, spec=spec)
        total += discount * score
        det = dict(det)
        det["step"] = step_idx
        det["discount"] = discount
        details.append(det)

        # advance state
        s1, _, _, _, _ = predict_fn(s, a)
        s = s1
        discount *= gamma

    return float(total), details, s

# ----------------------------- planning -----------------------------

def plan_and_decide(
    state: State,
    plan: Dict[str, Any],
    generate_candidates_fn: Callable[[State, Dict[str, Any]], List[Action]],
    predict_fn: Callable[[State, Action], Tuple[State, Latent, float, float, float]],
    *,
    r_ext: float = 0.0,
    energy_costs: Optional[Dict[str, float]] = None,
    spec: Optional[RewardSpec] = None,
    beam: int = 3,
    depth: int = 2,
    gamma: float = 0.9,
    method: str = "argmax",
) -> Tuple[Action, Dict[str, Any]]:
    """
    Orchestrates:
      1) generate base candidates
      2) build short sequences (len≤depth)
      3) score via short rollouts
      4) return the *first* action of the best sequence
    """
    spec = spec or get_default_spec()
    base_cands = generate_candidates_fn(state, plan) or [Action(kind="policy", name="ask_clarify", args={})]
    sequences = _default_seq_generator(state, plan, base_cands)

    # Simple beam pruning on sequence count (heuristic only)
    if beam > 0 and len(sequences) > beam:
        sequences = sequences[:beam]

    scored: List[Tuple[float, List[Dict[str, Any]], List[Action]]] = []
    for seq in sequences:
        seq_limited = seq[: max(1, min(len(seq), depth))]
        total, dets, _ = _rollout_sequence(
            state, seq_limited, predict_fn, r_ext=r_ext, spec=spec, energy_costs=energy_costs, gamma=gamma
        )
        scored.append((total, dets, seq_limited))

    if not scored:
        a = Action(kind="policy", name="ask_clarify", args={})
        return a, {"note": "no sequences", "chosen": [{"action": a.name, "score": 0.0}]}

    scored.sort(key=lambda t: t[0], reverse=True)
    best_total, best_details, best_seq = scored[0]
    best_action = best_seq[0]

    rationale = {
        "best_total": float(best_total),
        "best_seq": [
            {"name": d["action"]["name"], "score": float(d["shaped"]), "step": int(d["step"])}
            for d in best_details
        ],
        "alternatives": [
            {"total": float(tot), "seq": [d["action"]["name"] for d in dets]}
            for tot, dets, _ in scored[1:4]
        ],
        "intent": plan.get("intent", "unknown"),
        "depth_used": len(best_seq),
    }
    return best_action, rationale

if __name__ == "__main__":
    # quick self-test with fake predictors/generators
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

    def gen_cands(st: State, plan: Dict[str, Any]) -> List[Action]:
        it = plan.get("intent", "unknown")
        if it == "greeting":
            return [Action(kind="skill", name="reply_greeting", args={})]
        if it == "compute":
            return [Action(kind="tool", name="invoke_calc", args={"expr": "2+2"})]
        return [Action(kind="policy", name="ask_clarify", args={}), Action(kind="skill", name="reply_greeting", args={})]

    for it in ["greeting", "compute", "unknown"]:
        a, why = plan_and_decide(s0, {"intent": it}, gen_cands, fake_predict)
        print(f"[{it}] ->", a, "|", why)
