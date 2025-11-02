# -*- coding: utf-8 -*-
"""
NOEMA • safety/dsl.py — Minimal DSL for runtime safety rules (V0, stdlib-only + optional YAML)

Purpose
-------
Provide a tiny rule system to decide whether an action is "allow" / "block" / "review"
given a context: { text, plan, action, state, risk_base }.

Config format (YAML or JSON)
----------------------------
Example `config/safety.yaml`:

policies:
  - id: "deny_shell"
    desc: "Block any shell/system command"
    when:
      actions: ["run_shell", "call_os"]
      intents: ["compute", "unknown"]
      text_matches:
        deny:
          - "(?i)\\b(?:bash|sh|rm\\s+-rf|powershell|cmd\\.exe)\\b"
    effect: "block"     # "allow" | "block" | "review"
    risk_delta: 0.6
    message: "Shell execution is not allowed."

  - id: "math_safe"
    desc: "Allow safe arithmetic"
    when:
      actions: ["invoke_calc"]
      text_matches:
        allow:
          - "^[0-9+\\-*/() \\t]+$"
    effect: "allow"
    risk_delta: -0.2

  - id: "unknown_high_u"
    desc: "Review if intent unknown and uncertainty high"
    when:
      intents: ["unknown"]
      max_conf: 0.5
    effect: "review"
    risk_delta: 0.2

API
---
load_policies(path="config/safety.yaml") -> List[Rule]
evaluate(context, rules) -> Decision
safe_decide_allow(context, rules) -> (allowed: bool, reason: str, Decision)

Notes
-----
- If the config file is missing or invalid, no rules are loaded (default allow with risk_base).
- This module does not *execute* actions; it only suggests a decision.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import re
import json

# Optional YAML
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# ----------------------------- Data Types -----------------------------

@dataclass
class WhenClause:
    intents: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)      # action.name
    kinds:   List[str] = field(default_factory=list)      # action.kind
    text_allow: List[re.Pattern] = field(default_factory=list)
    text_deny : List[re.Pattern] = field(default_factory=list)
    min_conf: Optional[float] = None                      # match if conf ≥ min_conf
    max_conf: Optional[float] = None                      # match if conf ≤ max_conf
    max_u: Optional[float] = None                         # match if u ≤ max_u (uncertainty)

@dataclass
class Rule:
    id: str
    desc: str = ""
    effect: str = "allow"         # "allow" | "block" | "review"
    risk_delta: float = 0.0
    message: str = ""
    when: WhenClause = field(default_factory=WhenClause)

@dataclass
class Decision:
    effect: str                   # "allow" | "block" | "review"
    matched: List[str]
    risk: float                   # clipped to [0..1]
    reasons: List[str]
    details: Dict[str, Any] = field(default_factory=dict)

# ----------------------------- Helpers -----------------------------

_RE_FLAGS = re.IGNORECASE | re.MULTILINE

def _compile_patterns(spec: Dict[str, Any]) -> Tuple[List[re.Pattern], List[re.Pattern]]:
    allow: List[re.Pattern] = []
    deny:  List[re.Pattern] = []
    if not isinstance(spec, dict):
        return allow, deny
    for k, bucket in (("allow", allow), ("deny", deny)):
        pats = spec.get(k) or []
        for p in pats:
            try:
                bucket.append(re.compile(str(p), _RE_FLAGS))
            except Exception:
                continue
    return allow, deny

def _norm_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(v).strip().lower() for v in x if str(v).strip()]
    return [str(x).strip().lower()]

def _float_or_none(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None

# ----------------------------- Load Policies -----------------------------

def _parse_rule(d: Dict[str, Any]) -> Optional[Rule]:
    try:
        rid = str(d.get("id") or "").strip()
        if not rid:
            return None
        desc = str(d.get("desc") or "")
        effect = str(d.get("effect") or "allow").strip().lower()
        if effect not in ("allow", "block", "review"):
            effect = "allow"
        risk_delta = float(d.get("risk_delta", 0.0) or 0.0)
        message = str(d.get("message") or "")

        w = d.get("when") or {}
        intents = _norm_list(w.get("intents"))
        actions = _norm_list(w.get("actions"))
        kinds   = _norm_list(w.get("kinds"))
        t_allow, t_deny = _compile_patterns(w.get("text_matches") or {})
        min_conf = _float_or_none(w.get("min_conf"))
        max_conf = _float_or_none(w.get("max_conf"))
        max_u    = _float_or_none(w.get("max_u"))

        when = WhenClause(
            intents=intents, actions=actions, kinds=kinds,
            text_allow=t_allow, text_deny=t_deny,
            min_conf=min_conf, max_conf=max_conf, max_u=max_u
        )
        return Rule(id=rid, desc=desc, effect=effect, risk_delta=risk_delta, message=message, when=when)
    except Exception:
        return None

def load_policies(path: str | Path = "config/safety.yaml") -> List[Rule]:
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    data: Dict[str, Any] = {}
    if _HAS_YAML:
        try:
            data = yaml.safe_load(text) or {}
        except Exception:
            data = {}
    if not data:
        try:
            data = json.loads(text)
        except Exception:
            data = {}
    rules: List[Rule] = []
    for obj in (data.get("policies") or []):
        r = _parse_rule(obj or {})
        if r:
            rules.append(r)
    return rules

# ----------------------------- Evaluation -----------------------------

def _match_rule(ctx: Dict[str, Any], r: Rule) -> bool:
    plan = dict(ctx.get("plan") or {})
    intent = str(plan.get("intent") or "").strip().lower()
    action = dict(ctx.get("action") or {})
    a_name = str(action.get("name") or "").strip().lower()
    a_kind = str(action.get("kind") or "").strip().lower()
    text   = str(ctx.get("text") or "")

    # intent
    if r.when.intents and intent not in r.when.intents:
        return False
    # action
    if r.when.actions and a_name not in r.when.actions:
        return False
    # kind
    if r.when.kinds and a_kind not in r.when.kinds:
        return False
    # confidence / uncertainty constraints
    state = dict(ctx.get("state") or {})
    conf = state.get("conf", None)
    u    = state.get("u", None)
    if r.when.min_conf is not None and isinstance(conf, (int, float)):
        if float(conf) < float(r.when.min_conf):
            return False
    if r.when.max_conf is not None and isinstance(conf, (int, float)):
        if float(conf) > float(r.when.max_conf):
            return False
    if r.when.max_u is not None and isinstance(u, (int, float)):
        if float(u) > float(r.when.max_u):
            return False

    # deny patterns: if any match, the rule matches (effect typically "block")
    if r.when.text_deny:
        for pat in r.when.text_deny:
            if pat.search(text or ""):
                return True
        return False

    # allow patterns: if provided, at least one must match
    if r.when.text_allow:
        ok = any(pat.search(text or "") for pat in r.when.text_allow)
        return ok

    # no textual constraints → matched
    return True

def evaluate(context: Dict[str, Any], rules: List[Rule]) -> Decision:
    matched: List[Rule] = []
    for r in rules:
        try:
            if _match_rule(context, r):
                matched.append(r)
        except Exception:
            continue

    # accumulate risk
    risk = float(context.get("risk_base", 0.0) or 0.0)
    reasons: List[str] = []
    effects = set()
    for r in matched:
        risk += float(r.risk_delta or 0.0)
        effects.add(r.effect)
        if r.message:
            reasons.append(f"{r.id}: {r.message}")
        elif r.desc:
            reasons.append(f"{r.id}: {r.desc}")

    # effect priority: block > review > allow
    if "block" in effects:
        eff = "block"
    elif "review" in effects:
        eff = "review"
    else:
        eff = "allow"

    # clip risk
    risk = max(0.0, min(1.0, risk))

    return Decision(
        effect=eff,
        matched=[r.id for r in matched],
        risk=risk,
        reasons=reasons,
        details={"n_rules": len(rules), "n_matched": len(matched)},
    )

def safe_decide_allow(context: Dict[str, Any], rules: List[Rule]) -> Tuple[bool, str, Decision]:
    dec = evaluate(context, rules)
    allowed = (dec.effect == "allow")
    reason = " / ".join(dec.reasons) if dec.reasons else ("ok" if allowed else dec.effect)
    return allowed, reason, dec

if __name__ == "__main__":
    # Small self-check
    rules = [
        Rule(
            id="deny_shell", desc="no shell", effect="block", risk_delta=0.6,
            when=WhenClause(actions=["run_shell"], text_deny=[re.compile(r"\b(rm\s+-rf|bash|cmd\.exe)\b", _RE_FLAGS)])
        ),
        Rule(
            id="math_safe", desc="calc safe", effect="allow", risk_delta=-0.2,
            when=WhenClause(actions=["invoke_calc"], text_allow=[re.compile(r"^[0-9+\-*/() \t]+$")])
        ),
    ]
    ctx = {"text": "2+2", "plan": {"intent": "compute"}, "action": {"name": "invoke_calc", "kind": "tool"}}
    ok, reason, dec = safe_decide_allow(ctx, rules)
    print("allowed:", ok, "effect:", dec.effect, "reason:", reason)
