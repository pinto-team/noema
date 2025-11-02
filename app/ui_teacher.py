# -*- coding: utf-8 -*-
"""
NOEMA • app/ui_teacher.py — Teacher panel with 8 event tabs (Streamlit)

- Persists events as JSONL to logs/teacher_events.jsonl
- Event types: REINFORCE, CORRECT, DEMO, LABEL, PREF, TEST, RULE, CLARIFY
- Run: streamlit run app/ui_teacher.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import streamlit as st
import yaml

EVENTS_PATH = Path("logs/teacher_events.jsonl")
EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def write_event(ev: Dict) -> None:
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def base_event(ev_type: str, session_id: str, turn_id: Optional[str] = None) -> Dict:
    ev = {"type": ev_type, "session_id": session_id or "default", "ts": now_ts(), "payload": {}}
    if turn_id:
        ev["turn_id"] = turn_id
    return ev


st.set_page_config(page_title="NOEMA • Teacher Panel", layout="centered")
st.title("🧑‍🏫 NOEMA — Teacher Panel (V0)")

# ---- Sidebar / Session ----
with st.sidebar:
    st.markdown("### Session")
    session_id = st.text_input("session_id", value="S-LOCAL-001")
    turn_id_opt = st.text_input("turn_id (optional)", value="")
    st.caption("Each event will be stored with this session (and optional turn_id).")

    st.markdown("### Event log")
    st.code(str(EVENTS_PATH))
    if st.button("🔍 Show last 10 events"):
        if EVENTS_PATH.exists():
            tail = EVENTS_PATH.read_text(encoding="utf-8").splitlines()[-10:]
            st.text("\n".join(tail) if tail else "(empty)")
        else:
            st.info("No event file yet.")

tabs = st.tabs(
    ["REINFORCE", "CORRECT", "DEMO", "LABEL", "PREF", "TEST", "RULE", "CLARIFY"]
)

# ---- 1) REINFORCE ----
with tabs[0]:
    st.subheader("REINFORCE (+1 / 0 / −1)")
    val = st.select_slider("value", options=[-1, 0, 1], value=1)
    if st.button("✅ Submit REINFORCE"):
        ev = base_event("REINFORCE", session_id, turn_id_opt.strip() or None)
        ev["payload"]["value"] = val
        write_event(ev)
        st.success("Saved.")

# ---- 2) CORRECT ----
with tabs[1]:
    st.subheader("CORRECT (before → after)")
    before = st.text_input("before", value="Hello")
    after = st.text_input("after", value="Hello! Welcome")
    if st.button("✅ Submit CORRECT"):
        ev = base_event("CORRECT", session_id, turn_id_opt.strip() or None)
        ev["payload"].update({"before": before, "after": after})
        write_event(ev)
        st.success("Saved.")

# ---- 3) DEMO ----
with tabs[2]:
    st.subheader("DEMO (input → output)")
    inp = st.text_area("input", height=80, value="2+2?")
    out = st.text_area("output", height=80, value="4")
    if st.button("✅ Submit DEMO"):
        ev = base_event("DEMO", session_id, turn_id_opt.strip() or None)
        ev["payload"].update({"input": inp, "output": out})
        write_event(ev)
        st.success("Saved.")

# ---- 4) LABEL ----
with tabs[3]:
    st.subheader("LABEL (intent / entities)")
    intent = st.text_input("intent", value="greeting")
    entities = st.text_area("entities (JSON)", value="{}", height=100)
    labeled_input = st.text_area("input (optional, used for training)", value="", height=80)
    if st.button("✅ Submit LABEL"):
        try:
            ents = json.loads(entities) if entities.strip() else {}
            ev = base_event("LABEL", session_id, turn_id_opt.strip() or None)
            payload = {"intent": intent, "entities": ents}
            if labeled_input.strip():
                payload["input"] = labeled_input.strip()
            ev["payload"].update(payload)
            write_event(ev)
            st.success("Saved.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

# ---- 5) PREF ----
with tabs[4]:
    st.subheader("PREF (A ≻ B)")
    inp = st.text_area("input", height=80, value="Summarize this text:")
    A = st.text_area("A", height=80, value="A longer, somewhat scattered summary...")
    B = st.text_area("B", height=80, value="A shorter, clearer summary.")
    better = st.radio("better", options=["A", "B"], index=1, horizontal=True)
    if st.button("✅ Submit PREF"):
        ev = base_event("PREF", session_id, turn_id_opt.strip() or None)
        ev["payload"].update({"input": inp, "A": A, "B": B, "better": better})
        write_event(ev)
        st.success("Saved.")

# ---- 6) TEST ----
with tabs[5]:
    st.subheader("TEST (asserts)")
    name = st.text_input("name", value="arith-2plus2")
    t_input = st.text_area("input", height=60, value="2+2?")
    col1, col2 = st.columns(2)
    with col1:
        equals = st.text_input("assert.equals (optional)", value="4")
        contains = st.text_input("assert.contains (optional)", value="")
    with col2:
        latency = st.number_input(
            "assert.latency_ms_max (optional)", min_value=0, value=800, step=50
        )
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
        ev = base_event("TEST", session_id, turn_id_opt.strip() or None)
        ev["payload"].update({"name": name, "input": t_input, "assert": asserts})
        write_event(ev)
        st.success("Saved.")

# ---- 7) RULE ----
with tabs[6]:
    st.subheader("RULE (YAML spec)")
    rid = st.text_input("rule id", value="style.greeting")
    spec_text = st.text_area(
        "spec (YAML)", value="emoji_max: 1\ntone: polite\nlength_max: 80\n", height=140
    )
    if st.button("✅ Submit RULE"):
        try:
            spec = yaml.safe_load(spec_text) if spec_text.strip() else {}
            ev = base_event("RULE", session_id, turn_id_opt.strip() or None)
            ev["payload"].update({"id": rid, "spec": spec})
            write_event(ev)
            st.success("Saved.")
        except Exception as e:
            st.error(f"Invalid YAML: {e}")

# ---- 8) CLARIFY ----
with tabs[7]:
    st.subheader("CLARIFY (coach answer to agent's question)")
    question = st.text_input("Agent asked:", value="What do you mean by 'list'?")
    answer = st.text_input(
        "Coach answer:", value="Shopping list; sort by price ascending."
    )
    if st.button("✅ Submit CLARIFY"):
        ev = base_event("CLARIFY", session_id, turn_id_opt.strip() or None)
        ev["payload"].update({"question": question, "answer": answer})
        write_event(ev)
        st.success("Saved.")

st.markdown("---")
st.caption(
    "TIP: Events are appended in JSONL; the offline 'sleep' phase can consume them."
)
