# -*- coding: utf-8 -*-
"""
NOEMA • env/io_text.py — Minimal text I/O environment (V0, cleaned)

Responsibilities:
- Receive user text (begin_turn), deliver assistant output (deliver).
- Map trainer feedback {-1,0,+1} to r_ext in [-1.0..+1.0].
- Log an episode via memory.EpisodeStore when available; otherwise JSONL fallback.

Notes:
- Session id support (constructor arg).
- Safe import of EpisodeStore; no 'append' misuse.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
from pathlib import Path
import json, time

# ---------- Minimal action type (fallback if world is absent) ----------
try:
    from world import Action  # type: ignore
except Exception:
    @dataclass
    class Action:
        kind: str
        name: str
        args: Dict[str, Any]

# ---------- Episode logger: memory → else JSONL ----------
try:
    # Correct import path (matches memory/episodic.py)
    from memory.episodic import EpisodeStore  # type: ignore
    _HAS_MEM = True
except Exception:
    EpisodeStore = None  # type: ignore
    _HAS_MEM = False

class _JsonlLogger:
    """Fallback JSONL logger when EpisodeStore is unavailable."""
    def __init__(self, root: str | Path = "data/episodes"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # UI expects this exact filename in your app/ui_training_env.py
        self.fpath = self.root / "episodes.jsonl"

    def write(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self.fpath.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

# ---------- State / StepResult ----------

@dataclass
class IOState:
    turn_id: int = 0
    last_user_text: str = ""
    last_bot_text: str = ""
    ts: float = 0.0

@dataclass
class StepResult:
    text_out: str
    r_ext: float = 0.0
    done: bool = False
    info: Dict[str, Any] | None = None

# ---------- Text Environment ----------

class TextIOEnv:
    def __init__(self, episodes_root: str = "data/episodes", session_id: str = "S-LOCAL-001"):
        self.episodes_root = episodes_root
        self.session_id = str(session_id or "S-LOCAL-001")
        self.state = IOState()

        if _HAS_MEM and EpisodeStore is not None:
            self.store = EpisodeStore(episodes_root)  # type: ignore[call-arg]
            self._jsonl = None
        else:
            self.store = None
            self._jsonl = _JsonlLogger(episodes_root)

    # ---- Input side ----
    def reset(self) -> IOState:
        self.state = IOState(turn_id=0, last_user_text="", last_bot_text="", ts=time.time())
        return self.state

    def begin_turn(self, user_text: str) -> IOState:
        self.state.turn_id += 1
        self.state.last_user_text = str(user_text or "")
        self.state.ts = time.time()
        return self.state

    # ---- Helpers ----
    @staticmethod
    def _to_action_obj(action_like: Any) -> Action:
        if isinstance(action_like, Action):
            return action_like
        d = dict(action_like or {})
        return Action(
            kind=str(d.get("kind", "policy")),
            name=str(d.get("name", "ask_clarify")),
            args=dict(d.get("args", {})),
        )

    @staticmethod
    def _map_feedback_to_reward(feedback: Optional[int]) -> float:
        if feedback is None:
            return 0.0
        try:
            fb = int(feedback)
        except Exception:
            return 0.0
        if fb > 0:
            return +1.0
        if fb < 0:
            return -1.0
        return 0.0

    # ---- Output side + logging ----
    def deliver(
        self,
        *,
        intent: str,
        action: Any,
        text_out: str,
        meta: Optional[Dict[str, Any]] = None,
        feedback: Optional[int] = None,
        label_ok: Optional[bool] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        meta = dict(meta or {})
        extras = dict(extras or {})

        a = self._to_action_obj(action)
        self.state.last_bot_text = str(text_out or "")

        # External reward from trainer feedback
        r_ext = self._map_feedback_to_reward(feedback)

        # Pull optional signals
        conf = float(meta.get("confidence", meta.get("conf", 0.0) or 0.0))
        u = float(meta.get("u", (1.0 - conf) if conf else 0.0))
        r_total = float(meta.get("r_total", 0.0))
        risk = float(meta.get("risk", 0.0))
        r_int = float(meta.get("r_int", 0.0))
        energy = float(meta.get("energy", 0.0))

        # Optional vectors
        s_vec = list(meta.get("s_vec", []) or [])
        z_vec = list(meta.get("z_vec", []) or [])

        # ---- Write episode ----
        logged_ok = True
        if self.store is not None:
            try:
                # Use the standard EpisodeStore.log API (no unknown kwargs)
                self.store.log(
                    ts=time.time(),
                    session_id=self.session_id,
                    text_in=self.state.last_user_text,
                    text_out=self.state.last_bot_text,
                    intent=str(intent or "unknown"),
                    action_kind=a.kind,
                    action_name=a.name,
                    action_args=dict(a.args or {}),
                    r_total=r_total,
                    r_int=r_int,
                    r_ext=float(r_ext),
                    risk=risk,
                    energy=energy,
                    u=u,
                    conf=conf,
                    s_vec=s_vec,
                    z_vec=z_vec,
                    tests=list(extras.get("tests", []) or []),
                    costs=dict(extras.get("costs", {}) or {}),
                    tags=list(extras.get("tags", []) or []),
                )
            except Exception:
                logged_ok = False
        else:
            try:
                # Fallback JSONL schema (kept compatible with your UI)
                ep = {
                    "ts": time.time(),
                    "session_id": self.session_id,
                    "turn_id": int(self.state.turn_id),
                    "intent": str(intent or "unknown"),
                    "action_name": a.name,
                    "action_kind": a.kind,
                    "action_args": dict(a.args or {}),
                    "text_in": self.state.last_user_text,
                    "text_out": self.state.last_bot_text,
                    "conf": conf,
                    "u": u,
                    "r_total": r_total,
                    "r_int": r_int,
                    "r_ext": float(r_ext),
                    "risk": risk,
                    "energy": energy,
                    "label_ok": bool(label_ok) if (label_ok is not None) else None,
                    "z_vec": z_vec or None,
                    "s_vec": s_vec or None,
                    "extras": extras or None,
                }
                self._jsonl.write(ep)  # type: ignore[union-attr]
            except Exception:
                logged_ok = False

        info = {
            "session_id": self.session_id,
            "intent": str(intent or "unknown"),
            "action": {"name": a.name, "kind": a.kind, "args": dict(a.args or {})},
            "turn_id": int(self.state.turn_id),
            "conf": conf,
            "u": u,
            "r_total": r_total,
            "feedback": feedback,
            "logged": bool(logged_ok),
            # keep label_ok only in info to avoid EpisodeStore schema mismatch
            "label_ok": bool(label_ok) if (label_ok is not None) else None,
        }
        return StepResult(text_out=self.state.last_bot_text, r_ext=float(r_ext), done=False, info=info)

# ---- Self-test ----
if __name__ == "__main__":
    env = TextIOEnv()
    env.reset()
    env.begin_turn("سلام")
    out = env.deliver(
        intent="greeting",
        action={"kind": "skill", "name": "reply_greeting", "args": {}},
        text_out="سلام! خوش اومدی.",
        meta={"confidence": 0.9, "u": 0.1, "r_total": 0.6},
        feedback=+1,
    )
    print("OUT:", out.text_out, "| r_ext:", out.r_ext, "| info.logged:", out.info.get("logged"))
