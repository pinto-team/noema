# -*- coding: utf-8 -*-
"""
NOEMA • app/ui_training_env.py — Unified training UI (Chat + 8 teacher tools)

Run:
    streamlit run app/ui_training_env.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import yaml

from app.main import NoemaCore
from env import TextIOEnv, make_text_env

# ---------- Constants ----------
EVENTS_PATH = Path("logs/teacher_events.jsonl")
EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
def _now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds") + "Z"

def _write_event(ev: Dict) -> None:
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")

def _base_event(ev_type: str, session_id: str, turn_id: Optional[str] = None) -> Dict:
    ev = {"type": ev_type, "session_id": session_id or "default", "ts": _now_ts(), "payload": {}}
    if turn_id:
        ev["turn_id"] = turn_id
    return ev

def _ensure_state() -> None:
    ss = st.session_state
    ss.setdefault("episodes_root", "data/episodes/ui_training")
    ss.setdefault("session_id", "S-LOCAL-001")
    ss.setdefault("turn_id_opt", "")
    ss.setdefault("strict_mode", False)
    if "env" not in ss:
        env = make_text_env(ss["episodes_root"])
        env.reset()
        ss["env"] = env
    if "core" not in ss:
        ss["core"] = NoemaCore(strict=bool(ss["strict_mode"]))
    ss.setdefault("history", [])
    ss.setdefault("pending_reward", 0)

def _rebuild_core_env():
    ss = st.session_state
    env = make_text_env(ss["episodes_root"])
    env.reset()
    ss["env"] = env
    ss["core"] = NoemaCore(strict=bool(ss["strict_mode"]))
    ss["history"] = []
    ss["pending_reward"] = 0

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
    meta.update({"r_total": reward["r_total"], "r_int": reward["r_int"], "r_ext": reward["r_ext"], "risk": reward["risk"]})
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
    ss = st.session_state
    history: List[Dict[str, Any]] = ss.history
    if not history:
        return False
    last = history[-1]
    last["reward"] = int(feedback)
    if last.get("reward_logged"):
        return True
    payload = last.get("transition") or {}
    meta = dict(payload.get("meta") or {})
    extras = dict(payload.get("extras") or {}) or None
    env: TextIOEnv = ss.env
    try:
        env.deliver(
            intent=payload.get("intent", "unknown"),
            action=payload.get("action", {"kind": "policy", "name": "ask_clarify", "args": {}}),
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

# ---------- UI ----------
st.set_page_config(
    page_title="NOEMA • Unified Training UI",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🧠 NOEMA — Unified Training (Chat + Teacher Tools)")
_ensure_state()

# ----- Sidebar -----
with st.sidebar:
    st.header("Session")
    sid = st.text_input("session_id", value=st.session_state.session_id)
    if sid != st.session_state.session_id:
        st.session_state.session_id = sid
    tid = st.text_input("turn_id (optional, used in teacher events)", value=st.session_state.turn_id_opt)
    if tid != st.session_state.turn_id_opt:
        st.session_state.turn_id_opt = tid

    episodes_root_input = st.text_input("episodes_root", value=st.session_state.episodes_root)
    strict_mode = st.checkbox("Strict mode (require all modules; disable fallbacks)", value=bool(st.session_state.strict_mode))
    if episodes_root_input != st.session_state.episodes_root or strict_mode != bool(st.session_state.strict_mode):
        st.session_state.episodes_root = episodes_root_input
        st.session_state.strict_mode = bool(strict_mode)
        _rebuild_core_env()
        st.success("Settings applied. New session started.")

    st.markdown("---")
    st.caption("Environment log path:")
    st.code(str(Path(st.session_state.episodes_root) / "episodes.jsonl"))
    if st.button("🔍 Show last 10 environment logs"):
        env_path = Path(st.session_state.episodes_root) / "episodes.jsonl"
        if env_path.exists():
            tail = env_path.read_text(encoding="utf-8").splitlines()[-10:]
            st.text("\n".join(tail) if tail else "(empty)")
        else:
            st.info("Environment log file not found yet.")

    st.caption("Teacher events path:")
    st.code(str(EVENTS_PATH))
    if st.button("🔍 Show last 10 teacher events"):
        if EVENTS_PATH.exists():
            tail = EVENTS_PATH.read_text(encoding="utf-8").splitlines()[-10:]
            st.text("\n".join(tail) if tail else "(empty)")
        else:
            st.info("No event file yet.")

# ----- Main Tabs: Chat + Teacher Tools -----
tabs_main = st.tabs(["💬 Chat / Training", "🧑‍🏫 Teacher Tools (8)"])

# === Tab 1: Chat / Training ===
with tabs_main[0]:
    colL, colR = st.columns([3, 1])

    with colR:
        st.subheader("Coach reward")
        reward_choice = st.select_slider(
            "Reward for previous response",
            options=[-1, 0, 1],
            value=st.session_state.pending_reward,
            format_func=lambda v: {-1: "👎 -1", 0: "😐 0", 1: "👍 +1"}[v],
            key="reward_slider",
        )
        st.session_state.pending_reward = int(reward_choice)

        c1, c2, c3 = st.columns(3)
        if c1.button("👍 +1"):
            if _log_last_reply(1):
                st.session_state.pending_reward = 0
                st.rerun()
        if c2.button("😐 0"):
            if _log_last_reply(0):
                st.session_state.pending_reward = 0
                st.rerun()
        if c3.button("👎 -1"):
            if _log_last_reply(-1):
                st.session_state.pending_reward = 0
                st.rerun()

        st.markdown("---")
        if st.button("♻️ Start new session"):
            _rebuild_core_env()
            st.rerun()

    with colL:
        # Conversation history
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

        # Input area (chat_input)
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

                # Default turn_id field with env state
                st.session_state.turn_id_opt = str(getattr(env.state, "turn_id", ""))

                history.append(
                    {
                        "turn_id": getattr(env.state, "turn_id", len(history) + 1),
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

# === Tab 2: Teacher Tools (8) ===
with tabs_main[1]:
    st.markdown("از این تب می‌تونی هشت ابزار مربی را ثبت کنی. همه‌ی رویدادها در `logs/teacher_events.jsonl` ذخیره می‌شوند.")
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(
        ["REINFORCE", "CORRECT", "DEMO", "LABEL", "PREF", "TEST", "RULE", "CLARIFY"]
    )

    session_id = st.session_state.session_id
    turn_id_opt = st.session_state.turn_id_opt.strip() or None

    # 1) REINFORCE
    with t1:
        st.subheader("REINFORCE (+1 / 0 / −1)")
        val = st.select_slider("value", options=[-1, 0, 1], value=1)
        if st.button("✅ Submit REINFORCE"):
            ev = _base_event("REINFORCE", session_id, turn_id_opt)
            ev["payload"]["value"] = val
            _write_event(ev)
            st.success("Saved.")

    # 2) CORRECT
    with t2:
        st.subheader("CORRECT (before → after)")
        before = st.text_input("before", value="Hello")
        after = st.text_input("after", value="Hello! Welcome")
        if st.button("✅ Submit CORRECT"):
            ev = _base_event("CORRECT", session_id, turn_id_opt)
            ev["payload"].update({"before": before, "after": after})
            _write_event(ev)
            st.success("Saved.")

    # 3) DEMO
    with t3:
        st.subheader("DEMO (input → output)")
        inp = st.text_area("input", height=80, value="2+2?")
        out = st.text_area("output", height=80, value="4")
        if st.button("✅ Submit DEMO"):
            ev = _base_event("DEMO", session_id, turn_id_opt)
            ev["payload"].update({"input": inp, "output": out})
            _write_event(ev)
            st.success("Saved.")

    # 4) LABEL
    with t4:
        st.subheader("LABEL (intent / entities)")
        intent = st.text_input("intent", value="greeting")
        entities = st.text_area("entities (JSON)", value="{}", height=100)
        labeled_input = st.text_area("input (optional, used for training)", value="", height=80)
        if st.button("✅ Submit LABEL"):
            try:
                ents = json.loads(entities) if entities.strip() else {}
                ev = _base_event("LABEL", session_id, turn_id_opt)
                payload = {"intent": intent, "entities": ents}
                if labeled_input.strip():
                    payload["input"] = labeled_input.strip()
                ev["payload"].update(payload)
                _write_event(ev)
                st.success("Saved.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    # 5) PREF
    with t5:
        st.subheader("PREF (A ≻ B)")
        inp = st.text_area("input", height=80, value="Summarize this text:")
        A = st.text_area("A", height=80, value="A longer, somewhat scattered summary...")
        B = st.text_area("B", height=80, value="A shorter, clearer summary.")
        better = st.radio("better", options=["A", "B"], index=1, horizontal=True)
        if st.button("✅ Submit PREF"):
            ev = _base_event("PREF", session_id, turn_id_opt)
            ev["payload"].update({"input": inp, "A": A, "B": B, "better": better})
            _write_event(ev)
            st.success("Saved.")

    # 6) TEST
    with t6:
        st.subheader("TEST (asserts)")
        name = st.text_input("name", value="arith-2plus2")
        t_input = st.text_area("input", height=60, value="2+2?")
        col1, col2 = st.columns(2)
        with col1:
            equals = st.text_input("assert.equals (optional)", value="4")
            contains = st.text_input("assert.contains (optional)", value="")
        with col2:
            latency = st.number_input("assert.latency_ms_max (optional)", min_value=0, value=800, step=50)
            regex = st.text_input("assert.regex (optional)", value="")
        if st.button("✅ Submit TEST"):
            asserts = {}
            if equals.strip():
                asserts["equals"] = equals
            if contains.strip():
                asserts["contains"] = contains
            if latency > 0:
                asserts["latency_ms_max"] = int(latency)
            if regex.strip():
                asserts["regex"] = regex
            ev = _base_event("TEST", session_id, turn_id_opt)
            ev["payload"].update({"name": name, "input": t_input, "assert": asserts})
            _write_event(ev)
            st.success("Saved.")

    # 7) RULE
    with t7:
        st.subheader("RULE (YAML spec)")
        rid = st.text_input("rule id", value="style.greeting")
        spec_text = st.text_area("spec (YAML)", value="emoji_max: 1\ntone: polite\nlength_max: 80\n", height=140)
        if st.button("✅ Submit RULE"):
            try:
                spec = yaml.safe_load(spec_text) if spec_text.strip() else {}
                ev = _base_event("RULE", session_id, turn_id_opt)
                ev["payload"].update({"id": rid, "spec": spec})
                _write_event(ev)
                st.success("Saved.")
            except Exception as e:
                st.error(f"Invalid YAML: {e}")

    # 8) CLARIFY
    with t8:
        st.subheader("CLARIFY (coach answer to agent's question)")
        question = st.text_input("Agent asked:", value="What do you mean by 'list'?")
        answer = st.text_input("Coach answer:", value="Shopping list; sort by price ascending.")
        if st.button("✅ Submit CLARIFY"):
            ev = _base_event("CLARIFY", session_id, turn_id_opt)
            ev["payload"].update({"question": question, "answer": answer})
            _write_event(ev)
            st.success("Saved.")
