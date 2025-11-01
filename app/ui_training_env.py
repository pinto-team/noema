# -*- coding: utf-8 -*-
"""
NOEMA • app/ui_training_env.py — رابط آموزش تعاملی برای نوما (Streamlit)

ویژگی‌ها:
  - گفت‌وگوی زنده با هسته‌ی مینیمال نوما (`NoemaCore`).
  - امکان ثبت پاداش مربی برای پاسخ قبلی (−1 / 0 / +1).
  - ثبت هر گام در محیط متنی (`env.TextIOEnv`) همراه با متادیتا، تست‌ها و تصمیم.
  - مرور جزئیات هر گام (intent/action/پاداش‌ها/هزینه‌ها) برای مربیان و پژوهشگران.

اجرا:
    streamlit run app/ui_training_env.py

نکته: لاگ گام‌های محیط در `data/episodes/ui_training/episodes.jsonl` ذخیره می‌شود
      (در صورت نبود EpisodeStore). همچنین هسته‌ی نوما همچنان لاگ خود را در
      `logs/episodes.jsonl` به‌روزرسانی می‌کند.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from app.main import NoemaCore
from env import TextIOEnv, make_text_env


# ----- ابزارهای کمکی -----

def _now_ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


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
    action = {
        "kind": tr.a.kind,
        "name": tr.a.name,
        "args": dict(tr.a.args or {}),
    }
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
    meta.update({
        "r_total": reward["r_total"],
        "r_int": reward["r_int"],
        "r_ext": reward["r_ext"],
        "risk": reward["risk"],
    })

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
        st.error(f"ثبت در محیط با خطا روبه‌رو شد: {exc}")
        return False


def _reset_session() -> None:
    env = make_text_env(st.session_state.episodes_root)
    env.reset()
    st.session_state.env = env
    st.session_state.core = NoemaCore()
    st.session_state.history = []
    st.session_state.pending_reward = 0


# ----- رابط کاربری Streamlit -----

st.set_page_config(page_title="NOEMA • Training Environment", layout="wide")
st.title("🧠 NOEMA — محیط آموزش تعاملی")

_ensure_state()

st.markdown(
    """
این صفحه به مربی اجازه می‌دهد پاسخ‌های نوما را مشاهده کند، پاداش بیرونی (−1/0/+1)
ثبت کند و لاگ هر گام را در محیط آموزش ذخیره نماید. قبل از ارسال پیام تازه، مقدار
پاداش مربوط به پاسخ قبلی را از نوار کناری تعیین کنید.
"""
)

# ----- نوار کناری -----
with st.sidebar:
    st.header("تنظیمات جلسه")
    episodes_root_input = st.text_input(
        "episodes_root", value=st.session_state.episodes_root, key="episodes_root_input"
    )
    if episodes_root_input != st.session_state.episodes_root:
        st.session_state.episodes_root = episodes_root_input
        _reset_session()
        st.success("مسیر محیط به‌روزرسانی شد و جلسهٔ جدید آغاز شد.")

    reward_choice = st.select_slider(
        "پاداش برای پاسخ قبلی",
        options=[-1, 0, 1],
        value=st.session_state.pending_reward,
        format_func=lambda v: { -1: "−1 (نیاز به اصلاح)", 0: "۰ (خنثی)", 1: "+1 (عالی)" }[v],
        key="reward_slider",
    )
    st.session_state.pending_reward = int(reward_choice)

    if st.button("💾 ثبت پاداش پاسخ فعلی"):
        if st.session_state.history:
            if _log_last_reply(int(st.session_state.pending_reward)):
                st.session_state.pending_reward = 0
                st.rerun()
        else:
            st.info("هنوز پاسخی برای ارزیابی وجود ندارد.")

    if st.button("♻️ آغاز جلسهٔ تازه"):
        _reset_session()
        st.rerun()

    st.markdown("---")
    env_path = Path(st.session_state.episodes_root) / "episodes.jsonl"
    st.caption("مسیر لاگ محیط:")
    st.code(str(env_path))
    if st.button("🔍 نمایش آخرین ۱۰ لاگ محیط"):
        if env_path.exists():
            tail = env_path.read_text(encoding="utf-8").splitlines()[-10:]
            st.text("\n".join(tail) if tail else "(خالی)")
        else:
            st.info("فایل لاگ هنوز ایجاد نشده است.")

    core_log = Path("logs/episodes.jsonl")
    st.caption("مسیر لاگ داخلی نوما:")
    st.code(str(core_log))


# ----- نمایش گفت‌وگو -----
history: List[Dict[str, Any]] = st.session_state.history
for idx, step in enumerate(history):
    with st.chat_message("user"):
        st.markdown(step.get("user", ""))
        st.caption(f"Turn #{step.get('turn_id', idx + 1)} • {step.get('ts')}")

    with st.chat_message("assistant"):
        st.markdown(step.get("reply", ""))
        if step.get("reward_logged"):
            st.caption(f"پاداش مربی: {step.get('reward', 0):+d}")
        else:
            st.caption("منتظر پاداش مربی…")
        payload = step.get("transition") or {}
        with st.expander("جزئیات گام", expanded=False):
            st.write({
                "intent": payload.get("intent"),
                "action": payload.get("action"),
                "reward": payload.get("reward"),
                "meta": payload.get("meta"),
            })
            if payload.get("tests"):
                st.markdown("**Tests**")
                st.json(payload.get("tests"))
            if payload.get("costs"):
                st.markdown("**Costs**")
                st.json(payload.get("costs"))
            if payload.get("extras"):
                st.markdown("**Extras**")
                st.json(payload.get("extras"))


# ----- ورودی جدید مربی -----
user_message = st.chat_input("پیام مربی را بنویسید")
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
