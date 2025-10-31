# -*- coding: utf-8 -*-
"""
NOEMA • app/ui_teacher.py — پنل مربی ۸ تب (Streamlit)
- هر رویداد را به صورت JSONL در logs/teacher_events.jsonl ذخیره می‌کند.
- انواع رویداد: REINFORCE, CORRECT, DEMO, LABEL, PREF, TEST, RULE, CLARIFY
- اجرا: streamlit run app/ui_teacher.py
"""
from __future__ import annotations
import streamlit as st
import json, os, time, yaml
from datetime import datetime
from pathlib import Path

EVENTS_PATH = Path("logs/teacher_events.jsonl")
EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

def now_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def write_event(ev: dict) -> None:
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")

def base_event(ev_type: str, session_id: str) -> dict:
    return {"type": ev_type, "session_id": session_id or "default", "ts": now_ts(), "payload": {}}

st.set_page_config(page_title="NOEMA • Teacher Panel", layout="centered")
st.title("🧑‍🏫 NOEMA — Teacher Panel (V0)")

# ---- Header / Session ----
with st.sidebar:
    st.markdown("### Session")
    session_id = st.text_input("session_id", value="S-LOCAL-001")
    st.caption("هر رویداد با این شناسه ذخیره می‌شود.")
    st.markdown("### Event log")
    st.code(str(EVENTS_PATH))
    if st.button("🔍 نمایش آخرین ۱۰ رویداد"):
        if EVENTS_PATH.exists():
            tail = EVENTS_PATH.read_text(encoding="utf-8").splitlines()[-10:]
            st.text("\n".join(tail) if tail else "(خالی)")
        else:
            st.info("فایلی موجود نیست.")

tabs = st.tabs(["REINFORCE", "CORRECT", "DEMO", "LABEL", "PREF", "TEST", "RULE", "CLARIFY"])

# ---- 1) REINFORCE ----
with tabs[0]:
    st.subheader("REINFORCE (+1 / 0 / −1)")
    val = st.select_slider("value", options=[-1, 0, 1], value=1)
    if st.button("✅ ثبت REINFORCE"):
        ev = base_event("REINFORCE", session_id)
        ev["payload"]["value"] = val
        write_event(ev)
        st.success("ذخیره شد.")

# ---- 2) CORRECT ----
with tabs[1]:
    st.subheader("CORRECT (before → after)")
    before = st.text_input("before", value="سلام")
    after  = st.text_input("after",  value="سلام! خوش اومدی")
    if st.button("✅ ثبت CORRECT"):
        ev = base_event("CORRECT", session_id)
        ev["payload"].update({"before": before, "after": after})
        write_event(ev)
        st.success("ذخیره شد.")

# ---- 3) DEMO ----
with tabs[2]:
    st.subheader("DEMO (input → output)")
    inp = st.text_area("input", height=80, value="۲+۲؟")
    out = st.text_area("output", height=80, value="۴")
    if st.button("✅ ثبت DEMO"):
        ev = base_event("DEMO", session_id)
        ev["payload"].update({"input": inp, "output": out})
        write_event(ev)
        st.success("ذخیره شد.")

# ---- 4) LABEL ----
with tabs[3]:
    st.subheader("LABEL (intent / entities)")
    intent = st.text_input("intent", value="greeting")
    entities = st.text_area("entities (JSON)", value="{}", height=100)
    if st.button("✅ ثبت LABEL"):
        try:
            ents = json.loads(entities) if entities.strip() else {}
            ev = base_event("LABEL", session_id)
            ev["payload"].update({"intent": intent, "entities": ents})
            write_event(ev)
            st.success("ذخیره شد.")
        except Exception as e:
            st.error(f"JSON نامعتبر: {e}")

# ---- 5) PREF ----
with tabs[4]:
    st.subheader("PREF (A ≻ B)")
    inp = st.text_area("input", height=80, value="این متن را خلاصه کن:")
    A = st.text_area("A", height=80, value="خلاصه‌ی بلند و کمی پراکنده…")
    B = st.text_area("B", height=80, value="خلاصه‌ی کوتاه‌تر و شفاف.")
    better = st.radio("better", options=["A","B"], index=1, horizontal=True)
    if st.button("✅ ثبت PREF"):
        ev = base_event("PREF", session_id)
        ev["payload"].update({"input": inp, "A": A, "B": B, "better": better})
        write_event(ev)
        st.success("ذخیره شد.")

# ---- 6) TEST ----
with tabs[5]:
    st.subheader("TEST (asserts)")
    name = st.text_input("name", value="arith-2plus2")
    t_input = st.text_area("input", height=60, value="۲+۲؟")
    col1, col2 = st.columns(2)
    with col1:
        equals = st.text_input("assert.equals (اختیاری)", value="۴")
        contains = st.text_input("assert.contains (اختیاری)", value="")
    with col2:
        latency = st.number_input("assert.latency_ms_max (اختیاری)", min_value=0, value=800, step=50)
        regex = st.text_input("assert.regex (اختیاری)", value="")
    if st.button("✅ ثبت TEST"):
        asserts = {}
        if equals.strip():   asserts["equals"] = equals
        if contains.strip(): asserts["contains"] = contains
        if latency > 0:      asserts["latency_ms_max"] = int(latency)
        if regex.strip():    asserts["regex"] = regex
        ev = base_event("TEST", session_id)
        ev["payload"].update({"name": name, "input": t_input, "assert": asserts})
        write_event(ev)
        st.success("ذخیره شد.")

# ---- 7) RULE ----
with tabs[6]:
    st.subheader("RULE (YAML spec)")
    rid = st.text_input("rule id", value="style.greeting")
    spec_text = st.text_area(
        "spec (YAML)",
        value="emoji_max: 1\ntone: polite\nlength_max: 80\n",
        height=140
    )
    if st.button("✅ ثبت RULE"):
        try:
            spec = yaml.safe_load(spec_text) if spec_text.strip() else {}
            ev = base_event("RULE", session_id)
            ev["payload"].update({"id": rid, "spec": spec})
            write_event(ev)
            st.success("ذخیره شد.")
        except Exception as e:
            st.error(f"YAML نامعتبر: {e}")

# ---- 8) CLARIFY ----
with tabs[7]:
    st.subheader("CLARIFY (پاسخ روشن‌ساز)")
    question = st.text_input("نوما پرسیده:", value="منظورت از «لیست» چیه؟")
    answer   = st.text_input("پاسخ مربی:", value="لیست خرید؛ مرتب‌سازی بر اساس قیمت صعودی.")
    if st.button("✅ ثبت CLARIFY"):
        ev = base_event("CLARIFY", session_id)
        ev["payload"].update({"question": question, "answer": answer})
        write_event(ev)
        st.success("ذخیره شد.")

st.markdown("---")
st.caption("TIP: فایل رویدادها JSONL است؛ نوما می‌تواند این رویدادها را در فاز «خواب» مصرف کند.")
