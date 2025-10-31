# -*- coding: utf-8 -*-
"""
NOEMA • env/io_text.py — محیطِ متنیِ مینیمال (V0)

هدف:
  - یک «پوسته‌ی I/O متنی» برای نوما که ورودی کاربر (text_in) را دریافت
    و خروجی عامل (text_out) را ثبت کند.
  - سیگنال پاداش بیرونیِ مربی (feedback ∈ {−1,0,+1}) را به r_ext نگاشت می‌کند.
  - یک اپیزود سبک در لاگ می‌نویسد (اگر memory/EpisodeStore در دسترس بود از آن استفاده می‌کند؛
    وگرنه JSONL ساده).

API سریع:
    env = TextIOEnv(episodes_root="data/episodes")
    env.begin_turn("سلام")                      # ثبت ورودی کاربر
    step = env.deliver(
        intent="greeting",
        action={"kind":"skill","name":"reply_greeting","args":{}},
        text_out="سلام! خوش اومدی.",
        meta={"confidence":0.9, "u":0.1, "r_total":0.6},
        feedback=+1
    )
    print(step.text_out, step.r_ext)

قرارداد:
  - intent: "greeting" | "compute" | "unknown" | ...
  - action: دیکشنری سبک {kind,name,args} (یا شیء world.Action)
  - meta  : مقادیر اختیاری برای لاگ (conf/u/r_total/…)
  - feedback:  +1 / 0 / −1  (از طرف مربی؛ اختیاری)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
from pathlib import Path
import json, time

# ـــــــــــ انواع سبک (fallback اگر world موجود نبود) ـــــــــــ
try:
    from world import Action  # type: ignore
except Exception:
    @dataclass
    class Action:
        kind: str
        name: str
        args: Dict[str, Any]

# ـــــــــــ لاگر اپیزود: memory → در صورت نبود، JSONL ـــــــــــ
try:
    from memory import EpisodeStore  # type: ignore
    _HAS_MEM = True
except Exception:
    _HAS_MEM = False

class _JsonlLogger:
    """ثبت اپیزود در فایل JSONL اگر EpisodeStore موجود نباشد."""
    def __init__(self, root: str | Path = "data/episodes"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fpath = self.root / "episodes.jsonl"

    def write(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self.fpath.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

# ـــــــــــــــــــــــــــــ وضعیت/خروجی گام ـــــــــــــــــــــــــــــ

@dataclass
class IOState:
    turn_id: int = 0
    last_user_text: str = ""
    last_bot_text: str = ""
    ts: float = 0.0

@dataclass
class StepResult:
    text_out: str
    r_ext: float = 0.0            # پاداش بیرونی نگاشته‌شده به [-1..1]
    done: bool = False
    info: Dict[str, Any] = None   # جزئیات لاگ

# ـــــــــــــــــــــــــــــ محیط متنی ـــــــــــــــــــــــــــــ

class TextIOEnv:
    def __init__(self, episodes_root: str = "data/episodes"):
        self.episodes_root = episodes_root
        self.state = IOState()
        # لاگر
        if _HAS_MEM:
            self.store = EpisodeStore(episodes_root)
            self._jsonl = None
        else:
            self.store = None
            self._jsonl = _JsonlLogger(episodes_root)

    # ---------- آغاز/ورودی ----------

    def reset(self) -> IOState:
        """ریست ساده‌ی شمارنده/وضعیت."""
        self.state = IOState(turn_id=0, last_user_text="", last_bot_text="", ts=time.time())
        return self.state

    def begin_turn(self, user_text: str) -> IOState:
        """ثبت ورودی کاربر برای این نوبت."""
        self.state.turn_id += 1
        self.state.last_user_text = str(user_text or "")
        self.state.ts = time.time()
        return self.state

    # ---------- خروجی/ثبت ----------

    @staticmethod
    def _to_action_obj(action_like: Any) -> Action:
        if isinstance(action_like, Action):
            return action_like
        # انتظار dict با کلیدهای لازم
        d = dict(action_like or {})
        return Action(kind=str(d.get("kind","policy")), name=str(d.get("name","ask_clarify")), args=dict(d.get("args", {})))

    @staticmethod
    def _map_feedback_to_reward(feedback: Optional[int]) -> float:
        """
        +1 → +1.0   |   0/None → 0.0   |   -1 → -1.0
        می‌توانید در آینده شکل‌دهی ملایم اضافه کنید.
        """
        if feedback is None:
            return 0.0
        try:
            fb = int(feedback)
        except Exception:
            return 0.0
        if fb > 0: return +1.0
        if fb < 0: return -1.0
        return 0.0

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
        """
        خروجیِ عامل را می‌گیرد، r_ext را از feedback می‌سازد و اپیزود را ثبت می‌کند.
        """
        meta = dict(meta or {})
        extras = dict(extras or {})

        a = self._to_action_obj(action)
        self.state.last_bot_text = str(text_out or "")

        # سیگنال پاداش بیرونی
        r_ext = self._map_feedback_to_reward(feedback)

        # ساخت اپیزود
        ep = {
            "ts": time.time(),
            "turn_id": int(self.state.turn_id),
            "intent": str(intent or "unknown"),
            "action_name": a.name,
            "action_kind": a.kind,
            "text_in": self.state.last_user_text,
            "text_out": self.state.last_bot_text,
            # متریک‌های اختیاری
            "conf": float(meta.get("confidence", meta.get("conf", 0.0) or 0.0)),
            "u": float(meta.get("u", 1.0 - float(meta.get("confidence", 0.0))) or 0.0),
            "r_total": float(meta.get("r_total", 0.0)),
            "r_ext": float(r_ext),
            "risk": float(meta.get("risk", 0.0)),
            # برچسب/صحت اختیاری
            "label_ok": (True if label_ok else False) if (label_ok is not None) else meta.get("label_ok", None),
            # بردارهای نهان در صورت وجود
            "z_vec": meta.get("z_vec"),
            "s_vec": meta.get("s_vec"),
            # سایر
            "extras": extras if extras else None,
        }

        # نوشتن اپیزود
        try:
            if self.store is not None:
                self.store.append(ep)  # type: ignore[attr-defined]
            else:
                self._jsonl.write(ep)  # type: ignore[union-attr]
        except Exception as e:
            # شکست در لاگ نباید حلقه را بشکند
            pass

        info = {
            "intent": ep["intent"],
            "action": {"name": a.name, "kind": a.kind, "args": dict(a.args or {})},
            "turn_id": ep["turn_id"],
            "conf": ep["conf"],
            "u": ep["u"],
            "r_total": ep["r_total"],
            "feedback": feedback,
            "logged": True,
        }
        return StepResult(text_out=self.state.last_bot_text, r_ext=r_ext, done=False, info=info)

# ـــــــــــــــــــــــــــــ تست سریع/مستقیم ـــــــــــــــــــــــــــــ

if __name__ == "__main__":
    env = TextIOEnv()
    env.reset()
    env.begin_turn("سلام")
    out = env.deliver(
        intent="greeting",
        action={"kind":"skill","name":"reply_greeting","args":{}},
        text_out="سلام! خوش اومدی.",
        meta={"confidence":0.9, "u":0.1, "r_total":0.6},
        feedback=+1
    )
    print("OUT:", out.text_out, "| r_ext:", out.r_ext, "| info:", out.info)
