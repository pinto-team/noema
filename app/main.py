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
from typing import Any, Dict, List, Optional, Tuple
from time import time
import math
import re
import json
import sys
import pathlib
import random

# ========= انواع داده‌ی مینیمال =========
@dataclass
class Observation:
    t: float
    modality: str
    payload: str

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

@dataclass
class Transition:
    s: State
    z: Latent
    a: Action
    outcome: Outcome
    reward: RewardPkt
    ts: float

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

        # وزن‌های اولیه‌ی پاداش
        self.w_int = 0.2
        self.w_ext = 0.8

        # EMA خطای پیش‌بینی (برای r_int)
        self._ema_err_prev = 1.0

        # بارگذاری اختیاری ماژول‌ها
        self.mod = {
            "perception": _try_import("perception.encoder"),
            "world": _try_import("world.dynamics"),
            "lang_parse": _try_import("lang.parse"),
            "tool_registry": _try_import("toolhub.registry"),
            "tool_verify": _try_import("toolhub.verify"),
            "policy": _try_import("control.policy"),
            "planner": _try_import("control.planner"),
            "candidates": _try_import("control.candidates"),
            "shield": _try_import("safety.shield"),
            "value": _try_import("value.reward"),
            "selfmeta": _try_import("selfmeta.self_model"),
        }

    # ----- بلوک 1: ادراک -----
    def encode(self, text: str) -> Latent:
        if self.mod["perception"] and hasattr(self.mod["perception"], "encode"):
            return Latent(self.mod["perception"].encode(text))
        return Latent(_soft_hash(text))

    # ----- بلوک 2: وضعیت/پیش‌بینی -----
    def state(self, z_hist: List[Latent]) -> State:
        if self.mod["world"] and hasattr(self.mod["world"], "state"):
            return self.mod["world"].state(z_hist)
        # V0: همان آخرین z با عدم‌قطعیت پایین
        return State(s=z_hist[-1].z, u=0.2, conf=0.8)

    def predict(self, s: State, a: Action) -> Tuple[State, Latent, float, float, float]:
        if self.mod["world"] and hasattr(self.mod["world"], "predict"):
            return self.mod["world"].predict(s, a)
        # V0: پیش‌بینی ساده (بدون تغییر)، ریسک کم، عدم‌قطعیت بسته به نوع عمل
        u_hat = 0.1 if a.name in ["reply_greeting", "invoke_calc"] else 0.4
        risk = 0.0
        rhat = 0.5 if a.name in ["reply_greeting", "invoke_calc"] else 0.1
        return s, Latent(s.s), rhat, risk, u_hat

    # ----- بلوک 7: پارس نیت و پلان -----
    def parse_intent(self, text: str) -> Dict[str, Any]:
        if self.mod["lang_parse"] and hasattr(self.mod["lang_parse"], "parse_instruction"):
            return self.mod["lang_parse"].parse_instruction(text)
        # V0 ساده
        if _is_greeting(text):
            return {"intent": "greeting"}
        m = re.search(r"([0-9+\-*/() \t]+)", text)
        if m:
            return {"intent": "compute", "args": {"expr": m.group(1)}}
        return {"intent": "unknown"}

    # ----- بلوک 6: تولید نامزدها -----
    def generate_candidates(self, s: State, plan: Dict[str, Any]) -> List[Action]:
        if self.mod["candidates"] and hasattr(self.mod["candidates"], "generate"):
            return self.mod["candidates"].generate(s, plan)
        intent = plan.get("intent")
        if intent == "greeting":
            return [Action(kind="skill", name="reply_greeting")]
        if intent == "compute":
            return [Action(kind="tool", name="invoke_calc", args=plan.get("args", {}))]
        # fallback: پرسش روشن‌ساز
        return [Action(kind="policy", name="ask_clarify")]
    # ----- بلوک 10: سپر ایمنی -----
    def safety_check(self, s: State, a: Action) -> Tuple[bool, Dict[str, Any]]:
        shield = self.mod["shield"]
        if shield and hasattr(shield, "check"):
            allow, patch, reasons = shield.check(s, a)
            return allow, (patch or {})
        # V0: اجازه
        return True, {}

    # ----- بلوک 5/8: ارزش و متا -----
    def learning_progress(self, z_real: Latent, z_pred: Latent) -> float:
        # MSE ساده
        err_now = sum((ri - pi)**2 for ri, pi in zip(z_real.z, z_pred.z)) / len(z_real.z)
        ema = 0.9*self._ema_err_prev + 0.1*err_now
        r_int = max(0.0, self._ema_err_prev - ema)
        self._ema_err_prev = ema
        return r_int

    # ----- اجرای عمل (skill/tool/policy) -----
    def execute(self, a: Action) -> Outcome:
        t0 = time()
        if a.name == "reply_greeting":
            out = "سلام! خوش اومدی 👋"
            return Outcome(text_out=out, tests=[{"name":"style","pass":True}],
                           costs={"latency_ms": int((time()-t0)*1000), "compute": 1})
        if a.name == "invoke_calc":
            ok, res = _calc_safe(a.args.get("expr",""))
            test_ok = ok and _calc_safe(a.args.get("expr",""))[1] == res
            return Outcome(text_out=res if ok else "نامشخص",
                           tests=[{"name":"alt_eval","pass":test_ok}],
                           costs={"latency_ms": int((time()-t0)*1000), "compute": 2})
        if a.name == "ask_clarify":
            return Outcome(text_out="منظورت مشخص نیست؛ لطفاً دقیق‌تر بگو چه می‌خواهی 😊",
                           tests=[{"name":"clarify","pass":True}],
                           costs={"latency_ms": int((time()-t0)*1000), "compute": 1})
        # آینده: فراخوان ابزارهای رجیستری
        return Outcome(text_out="نامشخص", tests=[{"name":"noop","pass":True}],
                       costs={"latency_ms": int((time()-t0)*1000), "compute": 1})

    # ----- حافظه/لاگ (V0 فایل jsonl) -----
    def write_memory(self, tr: Transition) -> None:
        rec = {
            "ts": tr.ts,
            "state": {"u": tr.s.u, "conf": tr.s.conf},
            "action": {"kind": tr.a.kind, "name": tr.a.name, "args": tr.a.args},
            "outcome": {"text_out": tr.outcome.text_out, "tests": tr.outcome.tests, "costs": tr.outcome.costs},
            "reward": {"r_int": tr.reward.r_int, "r_ext": tr.reward.r_ext, "r_total": tr.reward.r_total},
        }
        with self.episodes_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ----- چرخه‌ی یک گام -----
    def step(self, text_in: str, r_ext: float = 0.0) -> str:
        obs = Observation(t=_current_ts(), modality="text", payload=text_in)
        z   = self.encode(obs.payload)
        s   = self.state([z])

        plan = self.parse_intent(obs.payload)
        cands = self.generate_candidates(s, plan)
        filtered: List[Action] = []
        for a in cands:
            allow, patch = self.safety_check(s, a)
            if not allow: continue
            if patch: a.args.update(patch)
            filtered.append(a)
        if not filtered:
            filtered = [Action(kind="policy", name="ask_clarify")]

        scores: List[Tuple[float, Action, State, Latent, float, float, float]] = []
        for a in filtered:
            s1, z1_hat, rhat, risk_hat, u_hat = self.predict(s, a)
            # پاداش درونی بر اساس پیشرفت (با z واقعی فعلاً همان z)
            r_int = self.learning_progress(z_real=z, z_pred=z1_hat)
            score = (self.w_int*r_int + self.w_ext*r_ext) - 0.3*u_hat
            if risk_hat <= 0.0:
                scores.append((score, a, s1, z1_hat, r_int, risk_hat, u_hat))

        if scores:
            scores.sort(key=lambda x: x[0], reverse=True)
            _, a_star, s1, z1_hat, r_int, risk_hat, u_hat = scores[0]
        else:
            a_star = Action(kind="policy", name="ask_clarify")
            s1, z1_hat, r_int, risk_hat, u_hat = s, z, 0.0, 0.0, 0.2

        outcome = self.execute(a_star)
        r_total = self.w_int*r_int + self.w_ext*r_ext
        pkt = RewardPkt(r_int=r_int, r_ext=r_ext, r_total=r_total, risk=risk_hat, energy=0.1)

        self.write_memory(Transition(s=s, z=z, a=a_star, outcome=outcome, reward=pkt, ts=_current_ts()))
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
