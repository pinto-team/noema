# -*- coding: utf-8 -*-
"""
NOEMA • app/main.py  —  مینیمال‌ترین حلقه‌ی اجرا برای V0
- این فایل مستقل اجرا می‌شود (بدون وابستگی به بقیه‌ی ماژول‌ها).
- اگر بعداً ماژول‌های تخصصی را ساختید (perception/world/…)، خودکار از آنها استفاده می‌کند.
- فعلاً دو توانمندی پایه دارد: پاسخ سلام و محاسبه‌ی ساده با ابزار داخلی + راستی‌آزمایی.
- TODOها را یکی‌یکی با فایل‌های شماره‌دار دیگر جایگزین کنید.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from time import time
import math
import re
import json
import sys
import pathlib
import random

# حافظه‌ی کاری اختیاری
try:
    from memory.wm import WorkingMemory  # type: ignore
except Exception:
    WorkingMemory = None  # type: ignore

# ========= انواع داده‌ی مینیمال =========
@dataclass
class Observation:
    t: float
    modality: str
    payload: str

try:  # تلاش برای استفاده از world.* در صورت موجود بودن
    from world import Latent, State, Action  # type: ignore
except Exception:  # fallback مینیمال
    @dataclass
    class Latent:
        z: List[float]

    @dataclass
    class State:
        s: List[float]
        u: float = 0.0   # uncertainty
        conf: float = 0.0

    @dataclass
    class Action:
        kind: str               # "skill" | "tool" | "policy"
        name: str               # e.g. "reply_greeting" | "invoke_calc"
        args: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RewardPkt:
    r_int: float
    r_ext: float
    r_total: float
    risk: float
    energy: float

@dataclass
class Outcome:
    text_out: Optional[str] = None
    tests: List[Dict[str, Any]] = field(default_factory=list)
    costs: Dict[str, Any] = field(default_factory=lambda: {"latency_ms": 0, "compute": 0})
    risk: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Transition:
    s: State
    z: Latent
    a: Action
    outcome: Outcome
    reward: RewardPkt
    ts: float
    plan: Dict[str, Any] = field(default_factory=dict)
    text_in: str = ""

# ========= کمک‌های داخلی (fallback) =========
def _soft_hash(text: str, d: int = 32) -> List[float]:
    """بردار سبک و پایدار از متن؛ صرفاً برای V0. با encoder واقعی جایگزین می‌شود."""
    random.seed(0)
    v = [0.0]*d
    for i,ch in enumerate(text):
        v[i % d] += (ord(ch) % 23) / 23.0
    # نرمال‌سازی
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

def _calc_safe(expr: str) -> Tuple[bool, str]:
    """ماشین‌حساب بسیار ساده و امن (فقط 0-9 + - * / ( ) و فاصله)."""
    if not re.fullmatch(r"[0-9+\-*/() \t]+", expr):
        return False, "invalid"
    try:
        # ارزیابی امن: فقط عملگرهای مجاز
        val = eval(expr, {"__builtins__": None}, {})
        return True, str(val)
    except Exception:
        return False, "error"

def _is_greeting(text: str) -> bool:
    t = text.strip().lower()
    return any(w in t for w in ["سلام", "درود", "hi", "hello", "hey"])

def _current_ts() -> float:
    return time()

# ========= تلاش برای بارگذاری ماژول‌های واقعی (اختیاری) =========
def _try_import(module: str, attr: Optional[str] = None):
    try:
        m = __import__(module, fromlist=['*'])
        return getattr(m, attr) if attr else m
    except Exception:
        return None

# ========= هسته‌ی مینیمال نوما =========

class NoemaCore:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_file = self.log_dir / "episodes.jsonl"

        # پیکربندی پایه
        self.energy_costs: Dict[str, float] = {
            "reply_greeting": 0.02,
            "invoke_calc": 0.05,
            "ask_clarify": 0.03,
        }
        self.state_window = 6
        self.decision_method = "argmax"

        # وزن‌های اولیه‌ی پاداش
        self.w_int = 0.2
        self.w_ext = 0.8

        # EMA خطای پیش‌بینی (برای r_int)
        self._ema_err_prev = 1.0

        # بارگذاری اختیاری توابع کلیدی
        self.modules: Dict[str, Any] = {
            "encode": _try_import("perception.encoder", "encode"),
            "state": _try_import("world", "state"),
            "predict": _try_import("world", "predict"),
            "parse": _try_import("lang.parse", "parse"),
            "detect_intent": _try_import("lang.parse", "detect_intent"),
            "load_style": _try_import("lang.format", "load_style"),
            "format_reply": _try_import("lang.format", "format_reply"),
            "candidates": _try_import("control.candidates", "generate"),
            "policy_decide": _try_import("control.policy", "decide"),
            "planner": _try_import("control.planner", "plan_and_decide"),
            "shield": _try_import("safety.shield", "check"),
            "reward_spec": _try_import("value.reward", "get_default_spec"),
            "reward_combine": _try_import("value.reward", "combine_rewards"),
            "reward_shape": _try_import("value.reward", "shape_bonus"),
            "reward_intrinsic": _try_import("value.reward", "intrinsic_from_errors"),
        }

        spec_fn = self.modules.get("reward_spec")
        self.reward_spec = None
        if callable(spec_fn):
            try:
                self.reward_spec = spec_fn()
            except Exception:
                self.reward_spec = None

        self.style = None
        style_loader = self.modules.get("load_style")
        if callable(style_loader):
            try:
                self.style = style_loader()
            except Exception:
                self.style = None

        skills_loader = _try_import("skills", "load_skills")
        self.skills = None
        if callable(skills_loader):
            try:
                self.skills = skills_loader()
            except Exception:
                self.skills = None

        tool_loader = _try_import("toolhub", "load_registry")
        self.tool_registry = None
        if callable(tool_loader):
            try:
                self.tool_registry = tool_loader()
            except Exception:
                self.tool_registry = None

        names_for_fallback: List[str] = []
        if self.skills:
            try:
                names_for_fallback.extend(self.skills.list_all())
            except Exception:
                pass
        names_for_fallback.extend(["reply_greeting", "invoke_calc"])
        self.skill_fallbacks: Dict[str, Callable[..., Dict[str, Any]]] = {}
        for name in dict.fromkeys(names_for_fallback):
            fn = _try_import(f"skills.{name}", "run")
            if callable(fn):
                self.skill_fallbacks[name] = fn

        self.wm = WorkingMemory(maxlen=32) if WorkingMemory else None
        self.last_decision: Dict[str, Any] = {}
        self.last_transition: Optional[Transition] = None
        self.last_outcome: Optional[Outcome] = None
        self.last_action: Optional[Action] = None
        self.last_plan: Dict[str, Any] = {}
        self.last_reward: Optional[RewardPkt] = None

    # ----- ابزارهای کمکی -----
    def _estimate_compute_cost(self, action_name: str) -> int:
        energy = float(self.energy_costs.get(action_name, 0.04))
        return max(1, int(round(energy * 50)))

    def _has_skill(self, name: str) -> bool:
        if self.skills is not None:
            try:
                if self.skills.has(name):
                    return True
            except Exception:
                pass
        return name in self.skill_fallbacks

    def _run_skill(
        self,
        name: str,
        obs: Observation,
        plan: Dict[str, Any],
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        params: Dict[str, Any] = dict(extra_args or {})
        params.setdefault("user_text", obs.payload)
        params.setdefault("plan", plan)
        if self.style is not None:
            params.setdefault("style", self.style)
        if self.tool_registry is not None and name == "invoke_calc":
            params.setdefault("tool_registry", self.tool_registry)
        plan_args = dict((plan or {}).get("args", {}) or {})
        for key, value in plan_args.items():
            params.setdefault(key, value)
        has_registry = False
        if self.skills is not None:
            try:
                has_registry = self.skills.has(name)
            except Exception:
                has_registry = False
        if has_registry:
            try:
                return self.skills.run(name, **params)
            except NotImplementedError:
                pass
            except Exception as exc:
                return {"error": str(exc)}
        fn = self.skill_fallbacks.get(name)
        if callable(fn):
            try:
                return fn(**params)
            except Exception as exc:
                return {"error": str(exc)}
        return None

    # ----- بلوک 1: ادراک -----
    def encode(self, text: str) -> Latent:
        fn = self.modules.get("encode")
        if callable(fn):
            try:
                return Latent(fn(text))
            except Exception:
                pass
        return Latent(_soft_hash(text))

    # ----- بلوک 2: وضعیت/پیش‌بینی -----
    def state(self, z_hist: List[Latent]) -> State:
        if not z_hist:
            return State(s=[], u=1.0, conf=0.0)
        fn = self.modules.get("state")
        if callable(fn):
            try:
                return fn(z_hist)
            except Exception:
                pass
        return State(s=z_hist[-1].z, u=0.2, conf=0.8)

    def predict(self, s: State, a: Action) -> Tuple[State, Latent, float, float, float]:
        fn = self.modules.get("predict")
        if callable(fn):
            try:
                return fn(s, a)
            except Exception:
                pass
        u_hat = 0.1 if a.name in ["reply_greeting", "invoke_calc"] else 0.4
        risk = 0.0
        rhat = 0.5 if a.name in ["reply_greeting", "invoke_calc"] else 0.1
        return s, Latent(s.s), rhat, risk, u_hat

    # ----- بلوک 7: پارس نیت و پلان -----
    def parse_intent(self, text: str) -> Dict[str, Any]:
        parse_fn = self.modules.get("parse")
        if callable(parse_fn):
            try:
                plan = parse_fn(text) or {}
                if isinstance(plan, dict):
                    return plan
            except Exception:
                pass
        detect_fn = self.modules.get("detect_intent")
        if callable(detect_fn):
            try:
                plan = detect_fn(text) or {}
                if isinstance(plan, dict):
                    return plan
            except Exception:
                pass
        if _is_greeting(text):
            return {"intent": "greeting", "args": {}, "confidence": 0.9}
        m = re.search(r"([0-9+\-*/() 	]+)", text)
        if m:
            expr = m.group(1)
            return {"intent": "compute", "args": {"expr": expr, "raw": text}, "confidence": 0.82}
        return {"intent": "unknown", "args": {"raw": text}, "confidence": 0.4}

    # ----- بلوک 6: تولید نامزدها -----
    def generate_candidates(self, s: State, plan: Dict[str, Any]) -> List[Action]:
        fn = self.modules.get("candidates")
        if callable(fn):
            try:
                return list(fn(s, plan, wm=self.wm, tool_registry=self.tool_registry))
            except TypeError:
                return list(fn(s, plan))
            except Exception:
                pass
        intent = plan.get("intent")
        args = dict((plan.get("args") or {}) or {})
        if intent == "greeting":
            return [Action(kind="skill", name="reply_greeting", args={})]
        if intent == "compute":
            expr = args.get("expr")
            cand_args = {"expr": expr} if isinstance(expr, str) and expr.strip() else {}
            return [Action(kind="tool", name="invoke_calc", args=cand_args)]
        return [Action(kind="policy", name="ask_clarify", args={})]

    # ----- بلوک 10: سپر ایمنی -----
    def safety_check(self, s: State, a: Action) -> Tuple[bool, Dict[str, Any]]:
        shield_fn = self.modules.get("shield")
        if callable(shield_fn):
            try:
                allow, patch, _ = shield_fn(s, a)
                return bool(allow), dict(patch or {})
            except Exception:
                pass
        return True, {}

    # ----- بلوک 5/8: ارزش و متا -----
    def learning_progress(self, z_real: Latent, z_pred: Latent) -> float:
        real = list(z_real.z or [])
        pred = list(z_pred.z or [])
        if not real or not pred:
            return 0.0
        err_now = sum((ri - pi) ** 2 for ri, pi in zip(real, pred)) / float(len(real))
        intrinsic = self.modules.get("reward_intrinsic")
        if callable(intrinsic):
            try:
                r_int, ema = intrinsic(self._ema_err_prev, err_now, alpha=0.9)
                self._ema_err_prev = float(ema)
                return float(r_int)
            except Exception:
                pass
        ema = 0.9 * self._ema_err_prev + 0.1 * err_now
        r_int = max(0.0, self._ema_err_prev - ema)
        self._ema_err_prev = ema
        return r_int

    # ----- اجرای عمل (skill/tool/policy) -----
    def execute(
        self,
        a: Action,
        obs: Observation,
        plan: Dict[str, Any],
        decision: Optional[Dict[str, Any]] = None,
    ) -> Outcome:
        t0 = time()
        compute_cost = self._estimate_compute_cost(a.name)
        decision = decision or {}
        intent = plan.get("intent", "unknown")
        if self._has_skill(a.name) or a.kind in {"skill", "tool"}:
            result = self._run_skill(a.name, obs, plan, extra_args=a.args)
            if isinstance(result, dict):
                meta = dict(result.get("meta") or {})
                if intent and "plan_intent" not in meta:
                    meta["plan_intent"] = intent
                if decision:
                    meta.setdefault("decision", decision)
                raw = result
                text_out = str(result.get("text_out") or result.get("output") or "")
                if not text_out and result.get("error"):
                    text_out = f"خطا در اجرای مهارت: {result['error']}"
                tests: List[Dict[str, Any]] = []
                if isinstance(result.get("tests"), list):
                    tests.extend(result["tests"])
                label_ok = result.get("label_ok")
                if label_ok is not None:
                    tests.append({"name": "label_ok", "pass": bool(label_ok)})
                elif not tests:
                    tests.append({"name": "skill_exec", "pass": result.get("error") is None})
                risk = float(meta.get("risk", 0.0))
                latency_ms = int((time() - t0) * 1000)
                return Outcome(
                    text_out=text_out or "نامشخص",
                    tests=tests,
                    costs={"latency_ms": latency_ms, "compute": compute_cost},
                    risk=risk,
                    meta=meta,
                    raw=raw,
                )

        meta: Dict[str, Any] = {"plan_intent": intent}
        if decision:
            meta["decision"] = decision
        tests: List[Dict[str, Any]] = []
        raw: Dict[str, Any] = {"action": a.name, "fallback": True}
        risk = 0.0
        if a.name == "reply_greeting":
            text_out = "سلام! خوش اومدی 👋"
            meta.setdefault("confidence", 0.9)
            tests.append({"name": "style", "pass": True})
        elif a.name == "invoke_calc":
            expr = None
            if isinstance(a.args, dict):
                expr = a.args.get("expr")
            if not expr:
                expr = (plan.get("args") or {}).get("expr")
            ok, res = _calc_safe(str(expr or ""))
            text_out = res if ok else "نامشخص"
            tests.append({"name": "safe_eval", "pass": ok})
            meta.setdefault("confidence", 0.82 if ok else 0.5)
            if not ok:
                meta["error"] = "invalid_expr"
        elif a.name == "ask_clarify":
            text_out = "منظورت مشخص نیست؛ لطفاً دقیق‌تر بگو چه می‌خواهی 😊"
            meta.setdefault("confidence", 0.55)
            meta.setdefault("u", 0.35)
            tests.append({"name": "clarify", "pass": True})
        else:
            text_out = "نامشخص"
            meta.setdefault("confidence", 0.4)
            tests.append({"name": "noop", "pass": True})
        latency_ms = int((time() - t0) * 1000)
        return Outcome(
            text_out=text_out,
            tests=tests,
            costs={"latency_ms": latency_ms, "compute": compute_cost},
            risk=risk,
            meta=meta,
            raw=raw,
        )

    # ----- حافظه/لاگ -----
    def write_memory(self, tr: Transition) -> None:
        rec = {
            "ts": tr.ts,
            "text_in": tr.text_in,
            "plan": tr.plan,
            "state": {"u": tr.s.u, "conf": tr.s.conf},
            "action": {"kind": tr.a.kind, "name": tr.a.name, "args": tr.a.args},
            "outcome": {
                "text_out": tr.outcome.text_out,
                "tests": tr.outcome.tests,
                "costs": tr.outcome.costs,
                "meta": tr.outcome.meta,
            },
            "reward": {
                "r_int": tr.reward.r_int,
                "r_ext": tr.reward.r_ext,
                "r_total": tr.reward.r_total,
                "risk": tr.reward.risk,
                "energy": tr.reward.energy,
            },
        }
        if tr.outcome.raw:
            rec["outcome"]["raw"] = tr.outcome.raw
        with self.episodes_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        if self.wm is not None:
            try:
                self.wm.push(
                    z=list(tr.z.z),
                    s=list(tr.s.s),
                    action={"kind": tr.a.kind, "name": tr.a.name, "args": dict(tr.a.args or {})},
                    outcome={"text_out": tr.outcome.text_out, "meta": tr.outcome.meta},
                    reward={"r_total": tr.reward.r_total, "r_int": tr.reward.r_int, "r_ext": tr.reward.r_ext},
                    text_in=tr.text_in,
                    ts=tr.ts,
                )
            except Exception:
                pass

    # ----- چرخه‌ی یک گام -----
    def step(self, text_in: str, r_ext: float = 0.0) -> str:
        obs = Observation(t=_current_ts(), modality="text", payload=text_in)
        z = self.encode(obs.payload)
        z_hist: List[Latent] = []
        if self.wm is not None:
            try:
                for vec in self.wm.z_hist(self.state_window - 1):
                    z_hist.append(Latent(list(vec)))
            except Exception:
                pass
        z_hist.append(z)
        s = self.state(z_hist)

        plan = self.parse_intent(obs.payload)
        if not isinstance(plan.get("args"), dict):
            plan["args"] = {}
        plan["args"].setdefault("raw", obs.payload)

        cands = self.generate_candidates(s, plan) or []
        filtered: List[Action] = []
        for cand in cands:
            allow, patch = self.safety_check(s, cand)
            if not allow:
                continue
            if patch:
                try:
                    cand.args.update(patch)
                except Exception:
                    cand.args = dict(cand.args or {})
                    cand.args.update(patch)
            filtered.append(cand)
        if not filtered:
            filtered = [Action(kind="policy", name="ask_clarify", args={})]

        policy_decide = self.modules.get("policy_decide")
        decision_details: List[Dict[str, Any]] = []
        a_star = filtered[0]
        if callable(policy_decide):
            try:
                a_star, decision_details = policy_decide(
                    s,
                    filtered,
                    self.predict,
                    r_ext=r_ext,
                    energy_costs=self.energy_costs,
                    spec=self.reward_spec,
                    method=self.decision_method,
                )
            except Exception:
                decision_details = []
        else:
            scores: List[Tuple[float, Action]] = []
            for cand in filtered:
                s1_tmp, z1_hat_tmp, _, _, u_hat_tmp = self.predict(s, cand)
                r_int_tmp = self.learning_progress(z_real=z, z_pred=z1_hat_tmp)
                score = (self.w_int * r_int_tmp + self.w_ext * r_ext) - 0.3 * u_hat_tmp
                scores.append((score, cand))
            scores.sort(key=lambda item: item[0], reverse=True)
            if scores:
                a_star = scores[0][1]
                decision_details = [
                    {
                        "action": {"kind": cand.kind, "name": cand.name, "args": cand.args},
                        "shaped": score,
                    }
                    for score, cand in scores[:3]
                ]
            else:
                a_star = Action(kind="policy", name="ask_clarify", args={})
                decision_details = []

        s1_pred, z1_hat, rhat_pred, risk_hat, u_hat = self.predict(s, a_star)
        r_int = self.learning_progress(z_real=z, z_pred=z1_hat)
        energy_cost = float(self.energy_costs.get(a_star.name, 0.05))
        conf1 = float(getattr(s1_pred, "conf", max(0.0, 1.0 - float(u_hat))))

        combine = self.modules.get("reward_combine")
        shape = self.modules.get("reward_shape")
        if callable(combine) and callable(shape) and self.reward_spec is not None:
            try:
                base = combine(r_int=r_int, r_ext=r_ext, risk=risk_hat, energy=energy_cost, spec=self.reward_spec)
                r_total = shape(base, confidence=conf1, u_hat=u_hat, spec=self.reward_spec)
            except Exception:
                r_total = self.w_int * r_int + self.w_ext * r_ext - 0.3 * u_hat
        else:
            r_total = self.w_int * r_int + self.w_ext * r_ext - 0.3 * u_hat

        decision_meta: Dict[str, Any] = {
            "predict": {
                "rhat": float(rhat_pred),
                "risk": float(risk_hat),
                "u_hat": float(u_hat),
                "conf": float(conf1),
            },
            "r_ext": float(r_ext),
            "r_int": float(r_int),
        }
        if decision_details:
            decision_meta["ranked"] = [
                {
                    "name": det.get("action", {}).get("name"),
                    "score": det.get("shaped"),
                    "rhat": det.get("rhat"),
                    "risk": det.get("risk"),
                    "u_hat": det.get("u_hat"),
                }
                for det in decision_details[:3]
            ]

        outcome = self.execute(a_star, obs, plan, decision=decision_meta)
        pkt = RewardPkt(r_int=r_int, r_ext=r_ext, r_total=r_total, risk=risk_hat, energy=energy_cost)
        transition = Transition(
            s=s,
            z=z,
            a=a_star,
            outcome=outcome,
            reward=pkt,
            ts=_current_ts(),
            plan=plan,
            text_in=obs.payload,
        )
        self.last_transition = transition
        self.last_outcome = outcome
        self.last_action = a_star
        self.last_plan = dict(plan)
        self.last_reward = pkt
        self.write_memory(transition)
        self.last_decision = decision_meta
        return outcome.text_out or ""

# ========= اجرای تعاملی کنسول =========
def main():
    core = NoemaCore()
    print("NOEMA V0 — آماده. (برای خروج Ctrl+C)\n")
    while True:
        try:
            text = input("شما: ").strip()
            if not text:
                continue
            # مربی اگر خواست پاداش سریع بدهد: +1 / 0 / -1 داخل براکت آخر پیام
            # مثال: "سلام [+1]"
            m = re.search(r"\[([+\-]?\d+)\]$", text)
            r_ext = 0.0
            if m:
                try:
                    r_ext = float(m.group(1))
                    text = text[:m.start()].strip()
                except Exception:
                    pass
            reply = core.step(text, r_ext=r_ext)
            print("نوما:", reply)
        except KeyboardInterrupt:
            print("\nخروج.")
            break
        except Exception as e:
            print("خطا:", e, file=sys.stderr)

if __name__ == "__main__":
    main()
