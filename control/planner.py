# -*- coding: utf-8 -*-
"""
NOEMA • control/planner.py — برنامه‌ریز کوتاه‌افق (V0، سبک و بدون وابستگی)
هدف:
  - از روی State فعلی + «طرح/نیت» (plan) و تابع پیش‌بینی world، چند دنباله‌ی کوتاه
    از اعمال می‌سازد (عمق 1..2)، امتیاز هر دنباله را محاسبه می‌کند و «اولین عمل»
    از بهترین دنباله را برمی‌گرداند.
  - امتیازدهی هر گام با control.policy.score_candidate انجام می‌شود (یک‌دست با سیاست).

API:
  plan_and_decide(
      state, plan, generate_candidates_fn, predict_fn,
      *, r_ext=0.0, beam=3, depth=2, gamma=0.9, method="argmax"
  ) -> (best_action, rationale)

یادداشت‌ها:
  - generate_candidates_fn همان چیزی‌ست که در app/main.py یا control/candidates.py دارید.
  - predict_fn همان world.predict است.
  - rationale شامل ریزجزئیات امتیازدهی دنباله‌ی برنده و چند دنباله‌ی جایگزین است.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

# انواع داده از world
try:
    from world import State, Latent, Action  # type: ignore
except Exception:
    from dataclasses import dataclass
    @dataclass
    class State:
        s: List[float]; u: float = 0.0; conf: float = 0.0
    @dataclass
    class Latent:
        z: List[float]
    @dataclass
    class Action:
        kind: str; name: str; args: Dict[str, Any]

# سیاست/امتیازدهی تک‌گام
from .policy import score_candidate, decide, RewardSpec, get_default_spec

# ----------------------------- توابع کمکی -----------------------------

def _default_seq_generator(
    state: State,
    plan: Dict[str, Any],
    base_candidates: List[Action],
) -> List[List[Action]]:
    """
    دنباله‌های پیشنهادی اولیه:
      - طول 1: همان نامزدهای پایه
      - طول 2: اگر نیت unknown → [ask_clarify → (هر نامزد پایه)]
               اگر intent=compute → [ask_clarify → invoke_calc] (به‌صورت احتیاطی)
    """
    seqs: List[List[Action]] = []
    # طول 1
    for a in base_candidates:
        seqs.append([a])

    intent = plan.get("intent", "unknown")
    if intent == "unknown":
        # clarify سپس بهترین حدسیات
        clarify = Action(kind="policy", name="ask_clarify", args={})
        for a in base_candidates:
            if a.name != "ask_clarify":
                seqs.append([clarify, a])
    elif intent == "compute":
        # در محاسبه: clarify→calc (اگر موجود باشد)
        clarify = Action(kind="policy", name="ask_clarify", args={})
        calc = None
        for a in base_candidates:
            if a.name == "invoke_calc":
                calc = a
                break
        if calc:
            seqs.append([clarify, calc])

    # حداکثر تنوع: حذف دنباله‌های تکراری
    uniq: List[List[Action]] = []
    seen = set()
    for seq in seqs:
        key = tuple((a.kind, a.name, tuple(sorted(a.args.items()))) for a in seq)
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
    امتیاز دنباله را با تنزیل γ محاسبه می‌کند و جزییات هر گام را برمی‌گرداند.
    """
    spec = spec or get_default_spec()
    total = 0.0
    details: List[Dict[str, Any]] = []
    s = state
    g = 1.0
    for step, a in enumerate(seq):
        score, det = score_candidate(s, a, predict_fn, r_ext=r_ext, energy_costs=energy_costs, spec=spec)
        total += g * score
        det = dict(det)
        det["step"] = step
        det["discount"] = g
        details.append(det)
        # حرکت وضعیت به s1 (از det مستقیم در دسترس نیست؛ دوباره predict می‌گیریم)
        s1, _, _, _, _ = predict_fn(s, a)
        s = s1
        g *= gamma
    return float(total), details, s

# ----------------------------- برنامه‌ریزی/انتخاب -----------------------------

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
    - ابتدا نامزدهای پایه را می‌سازد (generate_candidates_fn).
    - سپس دنباله‌های 1..depth را با ژنراتور پیش‌فرض می‌چیند (عمق 2 کافی است).
    - با rollout کوتاه، بهترین دنباله را برمی‌گزیند و «اولین عمل» را برمی‌گرداند.
    - rationale شامل: seq برنده، امتیاز کل، و چند seq جایگزین است.
    """
    spec = spec or get_default_spec()
    base_cands = generate_candidates_fn(state, plan) or [
        Action(kind="policy", name="ask_clarify", args={})
    ]

    # تولید دنباله‌ها (می‌توانید ژنراتور اختصاصی خودتان را این‌جا جایگزین کنید)
    seqs = _default_seq_generator(state, plan, base_cands)

    # Beam انتخابی (برای لیست‌های خیلی بزرگ)
    if beam > 0 and len(seqs) > beam:
        seqs = seqs[:beam] + sorted(seqs[beam:], key=lambda seq: -len(seq))[:max(0, beam//2)]

    scored: List[Tuple[float, List[Dict[str, Any]], List[Action]]] = []
    for seq in seqs:
        seq2 = seq[:max(1, min(len(seq), depth))]
        tot, dets, _ = _rollout_sequence(
            state, seq2, predict_fn, r_ext=r_ext, spec=spec, energy_costs=energy_costs, gamma=gamma
        )
        scored.append((tot, dets, seq2))

    if not scored:
        a = Action(kind="policy", name="ask_clarify", args={})
        return a, {"note": "no sequences", "chosen": [{"action": a.name, "score": 0.0}]}

    # مرتب‌سازی بر اساس امتیاز کل
    scored.sort(key=lambda t: t[0], reverse=True)
    best_total, best_details, best_seq = scored[0]
    best_action = best_seq[0]

    rationale = {
        "best_total": float(best_total),
        "best_seq": [{"name": d["action"]["name"], "score": d["shaped"], "step": d["step"]} for d in best_details],
        "alternatives": [
            {
                "total": float(tot),
                "seq": [d["action"]["name"] for d in dets]
            }
            for tot, dets, _ in scored[1:4]
        ],
        "intent": plan.get("intent", "unknown"),
    }
    return best_action, rationale

# ----------------------------- تست سریع -----------------------------

if __name__ == "__main__":
    # حالت و پیش‌بینی ساختگی
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

    def gen_cands(state: State, plan: Dict[str, Any]) -> List[Action]:
        intent = plan.get("intent", "unknown")
        if intent == "greeting":
            return [Action(kind="skill", name="reply_greeting", args={})]
        if intent == "compute":
            return [Action(kind="tool", name="invoke_calc", args={"expr": "2+2"})]
        return [
            Action(kind="policy", name="ask_clarify", args={}),
            Action(kind="skill",  name="reply_greeting", args={}),
        ]

    # سناریو ۱: greet
    a1, why1 = plan_and_decide(s0, {"intent":"greeting"}, gen_cands, fake_predict)
    print("best for greeting:", a1, "| rationale:", why1)

    # سناریو ۲: compute
    a2, why2 = plan_and_decide(s0, {"intent":"compute"}, gen_cands, fake_predict)
    print("best for compute:", a2, "| rationale:", why2)

    # سناریو ۳: unknown
    a3, why3 = plan_and_decide(s0, {"intent":"unknown"}, gen_cands, fake_predict)
    print("best for unknown:", a3, "| rationale:", why3)
