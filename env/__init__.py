# -*- coding: utf-8 -*-
"""
NOEMA • env package (V0)
- محیط‌های I/O برای نوما. در نسخه‌ی فعلی تنها محیط متنی سبک فراهم است.

استفاده‌ی سریع:
    from env import TextIOEnv, IOState, StepResult, make_text_env

    env = make_text_env(episodes_root="data/episodes")
    env.reset()
    env.begin_turn("سلام")
    step = env.deliver(
        intent="greeting",
        action={"kind":"skill","name":"reply_greeting","args":{}},
        text_out="سلام! خوش اومدی.",
        meta={"confidence":0.9, "u":0.1, "r_total":0.6},
        feedback=+1
    )
    print(step.text_out, step.r_ext)

یادداشت:
- اگر ماژول memory/* در دسترس باشد، لاگ به EpisodeStore می‌رود؛
  در غیر این صورت JSONL ساده در data/episodes/episodes.jsonl ثبت می‌شود.
"""

from __future__ import annotations

from .io_text import TextIOEnv, IOState, StepResult

__all__ = [
    "TextIOEnv",
    "IOState",
    "StepResult",
    "make_text_env",
]

def make_text_env(episodes_root: str = "data/episodes") -> TextIOEnv:
    """
    سازنده‌ی راحتِ محیط متن.
    """
    return TextIOEnv(episodes_root=episodes_root)

# اجرای کوتاه برای اطمینان از واردشدن بدون خطا
if __name__ == "__main__":
    env = make_text_env()
    env.reset()
    env.begin_turn("سلام")
    out = env.deliver(
        intent="greeting",
        action={"kind":"skill","name":"reply_greeting","args":{}},
        text_out="سلام! خوش اومدی.",
        meta={"confidence":0.9, "u":0.1, "r_total":0.6},
        feedback=+1
    )
    print("env ok — r_ext:", out.r_ext)
