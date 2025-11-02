# -*- coding: utf-8 -*-
"""
NOEMA • app/ui_training_env.py — Interactive training UI (Streamlit)

Features:
  - Live chat with NOEMA core (`NoemaCore`).
  - Record coach reward for the previous response (−1 / 0 / +1).
  - Log each step in the text environment (`env.TextIOEnv`) with metadata/tests/decisions.
  - Inspect intent/action/reward/costs per turn.

Run:
    streamlit run app/ui_training_env.py

Notes:
  - Environment logs are stored at `data/episodes/ui_training/episodes.jsonl`
    (when EpisodeStore is not present). Core logs are written to `logs/episodes.jsonl`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from app.main import NoemaCore
from env import TextIOEnv, make_text_env


# ----- Helpers -----
def _now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds") + "Z"


def _ensure_state() -> None:
    if "episodes_root" not in st.session_state:
        st.session_state.episodes_root = "data/episodes/ui_training"
    if "env" not in st.session_state:
        env = make_text_env(st.session_state.episodes_root)
        env.reset()
        st.session_state.env = env
    if "core" not in st.session_state:
        st.session_state.core = NoemaCore()
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, Any]] = []
    if "pending_reward" not in st.session_state:
        st.session_state.pending_reward = 0


def _build_transition_payload(core: NoemaCore) -> Optional[Dict[str, Any]]:
    tr = getattr(core, "last_transition", None)
    if tr is None:
        return None

    intent = str((tr.plan or {}).get("intent", "unknown"))
    action = {"kind": tr.a.kind, "name": tr.a.name, "args": dict(tr.a.args or {})}
    meta = dict(tr.outcome.meta or {})
    meta.setdefault("plan_intent", intent)
    if hasattr(tr.s, "conf"):
        meta.setdefault("confidence", float(getattr(tr.s, "conf")))
    if hasattr(tr.s, "u"):
        meta.setdefault("u", float(getattr(tr.s, "u")))

    reward = {
        "r_total": float(tr.reward.r_total),
        "r_int": float(tr.reward.r_int),
        "r_ext": float(tr.reward.r_ext),
        "risk": float(tr.reward.risk),
        "energy": float(tr.reward.energy),
    }
    meta.update(
        {
            "r_total": reward["r_total"],
            "r_int": reward["r_int"],
            "r_ext": reward["r_ext"],
            "risk": reward["risk"],
        }
    )

    tests = [dict(t) for t in tr.outcome.tests or []]
    costs = dict(tr.outcome.costs or {})

    extras: Dict[str, Any] = {
        "decision": getattr(core, "last_decision", {}),
        "tests": tests,
        "costs": costs,
        "plan": dict(tr.plan or {}),
    }
    extras = {k: v for k, v in extras.items() if v}

    label_ok = None
    if isinstance(tr.outcome.meta, dict) and "label_ok" in tr.outcome.meta:
        label_ok = tr.outcome.meta.get("label_ok")
    elif isinstance(tr.outcome.raw, dict) and "label_ok" in tr.outcome.raw:
        label_ok = tr.outcome.raw.get("label_ok")

    return {
        "intent": intent,
        "action": action,
        "meta": meta,
        "tests": tests,
        "costs": costs,
        "extras": extras,
        "reward": reward,
        "label_ok": label_ok,
        "plan": dict(tr.plan or {}),
    }


def _log_last_reply(feedback: int) -> bool:
    history: List[Dict[str, Any]] = st.session_state.history
    if not history:
        return False
    last = history[-1]
    last["reward"] = int(feedback)
    if last.get("reward_logged"):
        return True

    payload = last.get("transition") or {}
    meta = dict(payload.get("meta") or {})
    extras = dict(payload.get("extras") or {}) or None
    env: TextIOEnv = st.session_state.env

    try:
        env.deliver(
            intent=payload.get("intent", "unknown"),
            action=payload.get(
                "action", {"kind": "policy", "name": "ask_clarify", "args": {}}
            ),
            text_out=last.get("reply", ""),
            meta=meta,
            feedback=int(feedback),
            label_ok=payload.get("label_ok"),
            extras=extras,
        )
        last["reward_logged"] = True
        return True
    except Exception as exc:
        st.error(f"Environment logging failed: {exc}")
        return False


def _reset_session() -> None:
    env = make_text_env(st.session_state.episodes_root)
    env.reset()
    st.session_state.env = env
    st.session_state.core = NoemaCore()
    st.session_state.history = []
    st.session_state.pending_reward = 0


# ----- UI -----
st.set_page_config(page_title="NOEMA • Training Environment", layout="wide")
st.title("🧠 NOEMA — Interactive Training Environment")

_ensure_state()

st.markdown(
    """
This page lets the coach interact with NOEMA, assign external reward (−1/0/+1)
to the previous response, and log each step to the training environment.
Before sending a new message, set the reward for the last response in the sidebar.
"""
)

# ----- Sidebar -----
with st.sidebar:
    st.header("Session settings")
    episodes_root_input = st.text_input(
        "episodes_root", value=st.session_state.episodes_root, key="episodes_root_input"
    )
    if episodes_root_input != st.session_state.episodes_root:
        st.session_state.episodes_root = episodes_root_input
        _reset_session()
        st.success("Environment path updated; new session started.")

    reward_choice = st.select_slider(
        "Reward for previous response",
        options=[-1, 0, 1],
        value=st.session_state.pending_reward,
        format_func=lambda v: {-1: "-1 (needs fix)", 0: "0 (neutral)", 1: "+1 (great)"}[
            v
        ],
        key="reward_slider",
    )
    st.session_state.pending_reward = int(reward_choice)

    if st.button("💾 Save reward for current response"):
        if st.session_state.history:
            if _log_last_reply(int(st.session_state.pending_reward)):
                st.session_state.pending_reward = 0
                st.rerun()
        else:
            st.info("No response to evaluate yet.")

    if st.button("♻️ Start new session"):
        _reset_session()
        st.rerun()

    st.markdown("---")
    env_path = Path(st.session_state.episodes_root) / "episodes.jsonl"
    st.caption("Environment log path:")
    st.code(str(env_path))
    if st.button("🔍 Show last 10 environment logs"):
        if env_path.exists():
            tail = env_path.read_text(encoding="utf-8").splitlines()[-10:]
            st.text("\n".join(tail) if tail else "(empty)")
        else:
            st.info("Environment log file not found yet.")

    core_log = Path("logs/episodes.jsonl")
    st.caption("NOEMA core log path:")
    st.code(str(core_log))

# ----- Conversation view -----
history: List[Dict[str, Any]] = st.session_state.history
for idx, step in enumerate(history):
    with st.chat_message("user"):
        st.markdown(step.get("user", ""))
        st.caption(f"Turn #{step.get('turn_id', idx + 1)} • {step.get('ts')}")

    with st.chat_message("assistant"):
        st.markdown(step.get("reply", ""))
        if step.get("reward_logged"):
            st.caption(f"Coach reward: {step.get('reward', 0):+d}")
        else:
            st.caption("Awaiting coach reward…")
        payload = step.get("transition") or {}
        with st.expander("Step details", expanded=False):
            st.write(
                {
                    "intent": payload.get("intent"),
                    "action": payload.get("action"),
                    "reward": payload.get("reward"),
                    "meta": payload.get("meta"),
                }
            )
            if payload.get("tests"):
                st.markdown("**Tests**")
                st.json(payload.get("tests"))
            if payload.get("costs"):
                st.markdown("**Costs**")
                st.json(payload.get("costs"))
            if payload.get("extras"):
                st.markdown("**Extras**")
                st.json(payload.get("extras"))

# ----- New input -----
user_message = st.chat_input("Type your message")
if user_message is not None:
    text = user_message.strip()
    if text:
        core: NoemaCore = st.session_state.core
        env: TextIOEnv = st.session_state.env

        reward_to_pass = st.session_state.pending_reward if history else 0
        if history:
            _log_last_reply(int(st.session_state.pending_reward))

        env.begin_turn(text)
        reply = core.step(text, r_ext=float(reward_to_pass))
        payload = _build_transition_payload(core) or {}

        history.append(
            {
                "turn_id": env.state.turn_id,
                "user": text,
                "reply": reply,
                "ts": _now_ts(),
                "transition": payload,
                "reward": None,
                "reward_logged": False,
            }
        )

        st.session_state.pending_reward = 0
        st.rerun()
