# -*- coding: utf-8 -*-
"""
NOEMA • app/main.py — Minimal execution loop for V0 (with Strict Mode)

- Self-contained: runs even if optional modules are missing (non-strict).
- Strict mode (--strict): requires all real modules to be present & working; disables fallbacks.
- Provides two baseline capabilities out of the box (non-strict): greeting reply and safe arithmetic.
- All comments/strings are English; no language-specific dependency is required.
"""

from __future__ import annotations

import importlib
import json
import math
import pathlib
import random
import re
import sys
from dataclasses import dataclass, field, asdict
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional working memory
try:
    from memory.wm import WorkingMemory  # type: ignore
except Exception:
    WorkingMemory = None  # type: ignore

# Optional episodic store
try:
    from memory.episodic import EpisodeStore  # type: ignore
except Exception:
    EpisodeStore = None  # type: ignore

# Optional global workspace
try:
    from app.gws import GlobalWorkspace  # type: ignore
except Exception:
    GlobalWorkspace = None  # type: ignore

# Optional sleep cycle
try:
    from sleep.offline import SleepCfg, run_sleep_cycle  # type: ignore
except Exception:
    SleepCfg = None  # type: ignore
    run_sleep_cycle = None  # type: ignore


# ========= Minimal data types =========
@dataclass
class Observation:
    t: float
    modality: str
    payload: str


# Try importing world.*; otherwise fall back to light stubs (non-strict only)
try:
    from world import Latent, State, Action  # type: ignore
except Exception:

    @dataclass
    class Latent:
        z: List[float]

    @dataclass
    class State:
        s: List[float]
        u: float = 0.0  # uncertainty
        conf: float = 0.0

    @dataclass
    class Action:
        kind: str  # "skill" | "tool" | "policy"
        name: str  # e.g., "reply_greeting" | "invoke_calc"
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
    costs: Dict[str, Any] = field(
        default_factory=lambda: {"latency_ms": 0, "compute": 0}
    )
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
    focus: Optional[Dict[str, Any]] = None


# ========= REQUIRED symbols for Strict Mode =========
REQUIRED_SYMBOLS: Dict[str, List[str]] = {
    "perception.encoder": ["encode"],
    "world": ["state", "predict"],
    "lang.parse": ["parse"],  # one of parse|detect_intent required; checked separately
    "control.candidates": ["generate"],
    "control.policy": ["decide"],
    "safety.shield": ["check"],
    "value.reward": [
        "get_default_spec",
        "combine_rewards",
        "shape_bonus",
        "intrinsic_from_errors",
    ],
    "skills.reply_greeting": ["run"],
    "skills.invoke_calc": ["run"],
    "toolhub.registry": ["load_registry"],
    "memory.wm": ["WorkingMemory"],
    "memory.episodic": ["EpisodeStore"],
    "app.gws": ["GlobalWorkspace"],
    "sleep.offline": ["SleepCfg", "run_sleep_cycle"],
}


def _check_required(strict: bool) -> List[str]:
    """Return list of missing symbols when strict=True; empty otherwise."""
    if not strict:
        return []
    missing: List[str] = []
    for mod, names in REQUIRED_SYMBOLS.items():
        try:
            m = importlib.import_module(mod)
        except Exception:
            missing.append(f"{mod} (import)")
            continue
        for n in names:
            obj = getattr(m, n, None)
            ok = callable(obj) or (obj is not None and n[:1].isupper())
            if not ok:
                missing.append(f"{mod}.{n}")
    # at least one of parse|detect_intent must exist
    try:
        mp = importlib.import_module("lang.parse")
        if not (callable(getattr(mp, "parse", None)) or callable(getattr(mp, "detect_intent", None))):
            missing.append("lang.parse.parse|detect_intent (one required)")
    except Exception:
        missing.append("lang.parse (import)")
    return missing


# ========= Lightweight helpers (fallback; used only in non-strict or explicit) =========
def _soft_hash(text: str, d: int = 32) -> List[float]:
    """Create a stable lightweight vector from text (placeholder encoder for V0)."""
    random.seed(0)
    v = [0.0] * d
    for i, ch in enumerate(text):
        v[i % d] += (ord(ch) % 23) / 23.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _calc_safe(expr: str) -> Tuple[bool, str]:
    """Extremely safe arithmetic: only [0-9 + - * / ( ) whitespace]."""
    if not re.fullmatch(r"[0-9+\-*/() \t]+", expr):
        return False, "invalid"
    try:
        val = eval(expr, {"__builtins__": None}, {})
        return True, str(val)
    except Exception:
        return False, "error"


def _is_greeting(text: str) -> bool:
    """
    Language-agnostic heuristic for greetings (fallback only).
    Prefer dedicated NLP in lang.parse when available.
    """
    t = (text or "").strip().lower()
    tokens = (
        "hi",
        "hello",
        "hey",
        "hola",
        "hallo",
        "ciao",
        "salut",
        "bonjour",
        "namaste",
        "ola",
        "hei",
    )
    return any(tok in t for tok in tokens) and len(t) <= 40


def _current_ts() -> float:
    return time()


# ========= Best-effort import helper =========
def _try_import(module: str, attr: Optional[str] = None):
    try:
        m = __import__(module, fromlist=["*"])
        return getattr(m, attr) if attr else m
    except Exception:
        return None


# ========= NOEMA Core =========
class NoemaCore:
    def __init__(self, log_dir: str = "logs", strict: bool = False):
        self.strict = bool(strict)
        self.log_dir = pathlib.Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_file = self.log_dir / "episodes.jsonl"
        self.session_id = "S-CLI-LOCAL"
        self.episodes_root = pathlib.Path("data/episodes")
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        self.episode_store = None
        if EpisodeStore is not None:
            try:
                self.episode_store = EpisodeStore(self.episodes_root)
            except Exception:
                self.episode_store = None
        self.gws = GlobalWorkspace() if GlobalWorkspace is not None else None
        self.sleep_cfg = None
        if SleepCfg is not None:
            self.sleep_cfg = SleepCfg(
                events_path=str(self.log_dir / "teacher_events.jsonl"),
                episodes_root=str(self.episodes_root),
                write_rules=True,
                write_demos=True,
                build_tfidf=True,
                train_intent_clf=True,
            )
        self.last_focus: Optional[Dict[str, Any]] = None
        self.last_sleep_report: Optional[Dict[str, Any]] = None

        # Basic energy costs per action (approximate)
        self.energy_costs: Dict[str, float] = {
            "reply_greeting": 0.02,
            "invoke_calc": 0.05,
            "ask_clarify": 0.03,
        }
        self.state_window = 6
        self.decision_method = "argmax"

        # Reward weighting
        self.w_int = 0.2
        self.w_ext = 0.8

        # EMA of prediction error for intrinsic reward
        self._ema_err_prev = 1.0

        # Optional modules (best-effort imports)
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

        # In strict mode: verify presence of required symbols
        missing = _check_required(self.strict)
        if self.strict and missing:
            raise RuntimeError(
                "Strict mode: missing required symbols:\n  - " + "\n  - ".join(missing)
            )

        spec_fn = self.modules.get("reward_spec")
        self.reward_spec = None
        if callable(spec_fn):
            try:
                self.reward_spec = spec_fn()
            except Exception as exc:
                if self.strict:
                    raise
                self.reward_spec = None

        self.style = None
        style_loader = self.modules.get("load_style")
        if callable(style_loader):
            try:
                self.style = style_loader()
            except Exception:
                if self.strict:
                    raise
                self.style = None

        skills_loader = _try_import("skills", "load_skills")
        self.skills = None
        if callable(skills_loader):
            try:
                self.skills = skills_loader()
            except Exception:
                if self.strict:
                    raise
                self.skills = None

        tool_loader = _try_import("toolhub", "load_registry")
        self.tool_registry = None
        if callable(tool_loader):
            try:
                self.tool_registry = tool_loader()
            except Exception:
                if self.strict:
                    raise
                self.tool_registry = None

        names_for_fallback: List[str] = []
        if self.skills:
            try:
                names_for_fallback.extend(self.skills.list_all())
            except Exception:
                pass
        names_for_fallback.extend(
            ["reply_greeting", "invoke_calc", "reply_smalltalk", "reply_from_memory"]
        )
        self.skill_fallbacks: Dict[str, Callable[..., Dict[str, Any]]] = {}
        for name in dict.fromkeys(names_for_fallback):
            fn = _try_import(f"skills.{name}", "run")
            if callable(fn):
                self.skill_fallbacks[name] = fn

        # Disable fallbacks in strict mode (only real skill registry allowed)
        if self.strict:
            self.skill_fallbacks = {}

        self.wm = WorkingMemory(maxlen=32) if WorkingMemory else None
        self.last_decision: Dict[str, Any] = {}
        self.last_transition: Optional[Transition] = None
        self.last_outcome: Optional[Outcome] = None
        self.last_action: Optional[Action] = None
        self.last_plan: Dict[str, Any] = {}
        self.last_reward: Optional[RewardPkt] = None

    # ----- Cost estimation -----
    def _estimate_compute_cost(self, action_name: str) -> int:
        energy = float(self.energy_costs.get(action_name, 0.04))
        return max(1, int(round(energy * 50)))

    def _has_skill(self, name: str) -> bool:
        if self.skills is not None:
            try:
                if self.skills.has(name):
                    return True
            except Exception:
                if self.strict:
                    raise
        return (not self.strict) and (name in self.skill_fallbacks)

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
            except Exception as exc:
                if self.strict:
                    raise
                has_registry = False

        if has_registry:
            try:
                return self.skills.run(name, **params)
            except NotImplementedError:
                pass
            except Exception as exc:
                if self.strict:
                    raise
                return {"error": str(exc)}

        # fallback skills are disabled in strict mode
        fn = self.skill_fallbacks.get(name) if not self.strict else None
        if callable(fn):
            try:
                return fn(**params)
            except Exception as exc:
                if self.strict:
                    raise
                return {"error": str(exc)}
        return None

    # ----- Block 1: Perception/encoding -----
    def encode(self, text: str) -> Latent:
        fn = self.modules.get("encode")
        if callable(fn):
            try:
                return Latent(fn(text))
            except Exception:
                if self.strict:
                    raise
        # non-strict fallback
        return Latent(_soft_hash(text))

    # ----- Block 2: State/prediction -----
    def state(self, z_hist: List[Latent]) -> State:
        if not z_hist:
            return State(s=[], u=1.0, conf=0.0)
        fn = self.modules.get("state")
        if callable(fn):
            try:
                return fn(z_hist)
            except Exception:
                if self.strict:
                    raise
        return State(s=z_hist[-1].z, u=0.2, conf=0.8)

    def predict(self, s: State, a: Action) -> Tuple[State, Latent, float, float, float]:
        fn = self.modules.get("predict")
        if callable(fn):
            try:
                return fn(s, a)
            except Exception:
                if self.strict:
                    raise
        # Estimate predicted uncertainty and reward based on skill type
        if a.name in ["reply_greeting"]:
            u_hat, rhat = 0.1, 0.5
        elif a.name in ["invoke_calc"]:
            u_hat, rhat = 0.15, 0.6
        elif a.name in ["reply_from_memory"]:
            # Praise / memory replies are safe and valuable
            u_hat, rhat = 0.1, 0.8
        elif a.name in ["reply_smalltalk"]:
            u_hat, rhat = 0.25, 0.4
        elif a.name in ["ask_clarify"]:
            u_hat, rhat = 0.35, 0.2
        else:
            u_hat, rhat = 0.4, 0.1

        risk = 0.0

        return s, Latent(s.s), rhat, risk, u_hat

    # ----- Block 7: Intent parsing -----
    def parse_intent(self, text: str) -> Dict[str, Any]:
        parse_fn = self.modules.get("parse")
        if callable(parse_fn):
            try:
                plan = parse_fn(text) or {}
                if isinstance(plan, dict):
                    return plan
            except Exception:
                if self.strict:
                    raise

        detect_fn = self.modules.get("detect_intent")
        if callable(detect_fn):
            try:
                plan = detect_fn(text) or {}
                if isinstance(plan, dict):
                    return plan
            except Exception:
                if self.strict:
                    raise

        # Fallback parsing (non-strict only)
        if self.strict:
            raise RuntimeError("Strict mode: intent parser failed and fallback is disabled.")

        fa_digits = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
        ar_digits = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        ascii_math = (
            text.translate(fa_digits)
            .translate(ar_digits)
            .replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
        )

        if _is_greeting(text):
            return {"intent": "greeting", "args": {}, "confidence": 0.9}

        m = re.search(r"([0-9+\-*/() \t]+)", ascii_math)
        if m:
            expr = m.group(1)
            return {
                "intent": "compute",
                "args": {"expr": expr, "raw": text},
                "confidence": 0.82,
            }

        return {"intent": "smalltalk", "args": {"raw": text}, "confidence": 0.6}

    # ----- Block 6: Candidate generation -----
    def generate_candidates(self, s: State, plan: Dict[str, Any]) -> List[Action]:
        fn = self.modules.get("candidates")
        if callable(fn):
            try:
                return list(fn(s, plan, wm=self.wm, tool_registry=self.tool_registry))
            except TypeError:
                # backward-compat for older signatures
                try:
                    return list(fn(s, plan))
                except Exception:
                    if self.strict:
                        raise
            except Exception:
                if self.strict:
                    raise

        if self.strict:
            raise RuntimeError("Strict mode: candidate generator failed and fallback is disabled.")

        intent = plan.get("intent")
        args = dict((plan.get("args") or {}) or {})
        if intent == "greeting":
            return [Action(kind="skill", name="reply_greeting", args={})]
        if intent == "compute":
            expr = args.get("expr")
            cand_args = {"expr": expr} if isinstance(expr, str) and expr.strip() else {}
            return [Action(kind="tool", name="invoke_calc", args=cand_args)]
        if intent == "smalltalk":
            return [Action(kind="skill", name="reply_smalltalk", args={})]
        return [Action(kind="skill", name="reply_from_memory", args={})]

    # ----- Block 10: Safety -----
    def safety_check(self, s: State, a: Action) -> Tuple[bool, Dict[str, Any]]:
        shield_fn = self.modules.get("shield")
        if callable(shield_fn):
            try:
                allow, patch, _ = shield_fn(s, a)
                return bool(allow), dict(patch or {})
            except Exception:
                if self.strict:
                    raise
        # non-strict default: allow
        return True, {}

    # ----- Block 5/8: Value & meta -----
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
                if self.strict:
                    raise

        ema = 0.9 * self._ema_err_prev + 0.1 * err_now
        r_int = max(0.0, self._ema_err_prev - ema)
        self._ema_err_prev = ema
        return r_int

    # ----- Execution (skill/tool/policy) -----
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

        # Strict guard: if an external skill/tool is requested but not available, raise
        if self.strict and a.kind in {"skill", "tool"} and not self._has_skill(a.name):
            raise RuntimeError(f"Strict mode: skill/tool '{a.name}' not available.")

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
                    text_out = f"Error executing skill: {result['error']}"
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
                    text_out=text_out or "unknown",
                    tests=tests,
                    costs={"latency_ms": latency_ms, "compute": compute_cost},
                    risk=risk,
                    meta=meta,
                    raw=raw,
                )

        # Non-strict fallback execution for basic actions
        if self.strict:
            raise RuntimeError(f"Strict mode: execution fallback reached for '{a.name}'.")

        meta: Dict[str, Any] = {"plan_intent": intent}
        if decision:
            meta["decision"] = decision
        tests: List[Dict[str, Any]] = []
        raw: Dict[str, Any] = {"action": a.name, "fallback": True}
        risk = 0.0

        if a.name == "reply_greeting":
            text_out = "Hello!"
            meta.setdefault("confidence", 0.9)
            tests.append({"name": "style", "pass": True})
        elif a.name == "invoke_calc":
            expr = None
            if isinstance(a.args, dict):
                expr = a.args.get("expr")
            if not expr:
                expr = (plan.get("args") or {}).get("expr")
            ok, res = _calc_safe(str(expr or ""))
            text_out = res if ok else "unknown"
            tests.append({"name": "safe_eval", "pass": ok})
            meta.setdefault("confidence", 0.82 if ok else 0.5)
            if not ok:
                meta["error"] = "invalid_expr"
        elif a.name == "ask_clarify":
            text_out = "Please clarify your intent."
            meta.setdefault("confidence", 0.55)
            meta.setdefault("u", 0.35)
            tests.append({"name": "clarify", "pass": True})
        else:
            text_out = "unknown"
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

    # ----- Memory/logging -----
    def write_memory(self, tr: Transition) -> None:
        focus = tr.focus or None
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
        if focus:
            rec["focus"] = focus
        state_vec = []
        try:
            state_vec = list(getattr(tr.s, "s", []) or [])
        except Exception:
            state_vec = []
        latent_vec = []
        try:
            latent_vec = list(getattr(tr.z, "z", []) or [])
        except Exception:
            latent_vec = []
        if state_vec:
            rec["state"]["s_vec"] = state_vec
        if latent_vec:
            rec["latent"] = {"z": latent_vec}

        if self.episode_store is not None:
            try:
                tags = list(tr.plan.get("tags", []) or [])
                mode = str(focus.get("mode")) if focus else ""
                if mode:
                    tags.append(f"focus:{mode}")
                self.episode_store.log_transition(
                    session_id=self.session_id,
                    text_in=tr.text_in,
                    state={"u": tr.s.u, "conf": tr.s.conf, "s": state_vec},
                    latent={"z": latent_vec},
                    action={"kind": tr.a.kind, "name": tr.a.name, "args": dict(tr.a.args or {})},
                    outcome={
                        "text_out": tr.outcome.text_out,
                        "tests": tr.outcome.tests,
                        "costs": tr.outcome.costs,
                        "meta": tr.outcome.meta,
                        "intent": tr.plan.get("intent"),
                        "focus": focus,
                    },
                    reward={
                        "r_int": tr.reward.r_int,
                        "r_ext": tr.reward.r_ext,
                        "r_total": tr.reward.r_total,
                        "risk": tr.reward.risk,
                        "energy": tr.reward.energy,
                    },
                    ts=tr.ts,
                    tags=tags,
                )
            except Exception:
                if self.strict:
                    raise
        with self.episodes_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        if self.wm is not None:
            try:
                self.wm.push(
                    z=list(latent_vec),
                    s=list(state_vec),
                    action={
                        "kind": tr.a.kind,
                        "name": tr.a.name,
                        "args": dict(tr.a.args or {}),
                    },
                    outcome={"text_out": tr.outcome.text_out, "meta": tr.outcome.meta},
                    reward={
                        "r_total": tr.reward.r_total,
                        "r_int": tr.reward.r_int,
                        "r_ext": tr.reward.r_ext,
                    },
                    text_in=tr.text_in,
                    ts=tr.ts,
                )
            except Exception:
                if self.strict:
                    raise

    def _maybe_run_sleep_cycle(self) -> None:
        if self.gws is None or self.sleep_cfg is None or run_sleep_cycle is None:
            return
        if not self.gws.should_sleep_now():
            return
        try:
            report = run_sleep_cycle(self.sleep_cfg, verbose=False)
            self.last_sleep_report = report
            self.gws.mark_slept()
        except Exception:
            # Sleep is best-effort even in strict mode
            pass

    # ----- One-step loop -----
    def step(self, text_in: str, r_ext: float = 0.0) -> str:
        obs = Observation(t=_current_ts(), modality="text", payload=text_in)
        z = self.encode(obs.payload)

        z_hist: List[Latent] = []
        if self.wm is not None:
            try:
                for vec in self.wm.z_hist(self.state_window - 1):
                    z_hist.append(Latent(list(vec)))
            except Exception:
                if self.strict:
                    raise
        z_hist.append(z)

        s = self.state(z_hist)
        plan = self.parse_intent(obs.payload)
        if not isinstance(plan.get("args"), dict):
            plan["args"] = {}
        plan["args"].setdefault("raw", obs.payload)

        focus_dict: Optional[Dict[str, Any]] = None
        if self.gws is not None:
            try:
                focus_obj = self.gws.tick(
                    text=obs.payload,
                    intent=str(plan.get("intent", "unknown")),
                    model_signals={
                        "u_mean": float(getattr(s, "u", 0.0)),
                        "risk": float(getattr(self.last_reward, "risk", 0.0)) if self.last_reward else 0.0,
                    },
                )
                if focus_obj is not None:
                    if hasattr(focus_obj, "__dataclass_fields__"):
                        focus_dict = asdict(focus_obj)
                    elif isinstance(focus_obj, dict):
                        focus_dict = dict(focus_obj)
            except Exception:
                if self.strict:
                    raise
                focus_dict = None
        self.last_focus = focus_dict

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
                if self.strict:
                    raise
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
                        "action": {
                            "kind": cand.kind,
                            "name": cand.name,
                            "args": cand.args,
                        },
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
                base = combine(
                    r_int=r_int, r_ext=r_ext, risk=risk_hat, energy=energy_cost, spec=self.reward_spec
                )
                r_total = shape(base, confidence=conf1, u_hat=u_hat, spec=self.reward_spec)
            except Exception:
                if self.strict:
                    raise
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
        if focus_dict:
            decision_meta["focus"] = focus_dict

        outcome = self.execute(a_star, obs, plan, decision=decision_meta)
        if focus_dict:
            try:
                outcome.meta.setdefault("focus", focus_dict)
                outcome.meta.setdefault("gws_mode", focus_dict.get("mode"))
            except Exception:
                if self.strict:
                    raise
        pkt = RewardPkt(
            r_int=r_int, r_ext=r_ext, r_total=r_total, risk=risk_hat, energy=energy_cost
        )
        transition = Transition(
            s=s,
            z=z,
            a=a_star,
            outcome=outcome,
            reward=pkt,
            ts=_current_ts(),
            plan=plan,
            text_in=obs.payload,
            focus=focus_dict,
        )
        self.last_transition = transition
        self.last_outcome = outcome
        self.last_action = a_star
        self.last_plan = dict(plan)
        self.last_reward = pkt
        self.write_memory(transition)
        self.last_decision = decision_meta
        self._maybe_run_sleep_cycle()
        return outcome.text_out or ""


# ========= CLI loop =========
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--strict", action="store_true", help="Require all modules; disable fallbacks.")
    p.add_argument("--log-dir", type=str, default="logs", help="Logs directory (default: logs)")
    args = p.parse_args()

    core = NoemaCore(log_dir=args.log_dir, strict=args.strict)
    print(f"NOEMA V0 — Ready. Strict: {args.strict}  (Ctrl+C to exit)\n")
    while True:
        try:
            text = input("User: ").strip()
            if not text:
                continue
            # Optional inline reward for previous response: +1 / 0 / -1 in square brackets at the end
            # Example: "Hello [+1]"
            m = re.search(r"\[([+\-]?\d+)\]$", text)
            r_ext = 0.0
            if m:
                try:
                    r_ext = float(m.group(1))
                    text = text[: m.start()].strip()
                except Exception:
                    pass
            reply = core.step(text, r_ext=r_ext)
            print("Noema:", reply)
        except KeyboardInterrupt:
            print("\nExit.")
            break
        except Exception as e:
            print("Error:", e, file=sys.stderr)
            if args.strict:
                # in strict mode, fail fast
                break


if __name__ == "__main__":
    main()
