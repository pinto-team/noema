# -*- coding: utf-8 -*-
"""
NOEMA • env package (V0)
- Text I/O environments for NOEMA.
"""

from __future__ import annotations

from .io_text import TextIOEnv, IOState, StepResult

__all__ = [
    "TextIOEnv",
    "IOState",
    "StepResult",
    "make_text_env",
]

def make_text_env(episodes_root: str = "data/episodes", session_id: str = "S-LOCAL-001") -> TextIOEnv:
    """Convenience factory."""
    return TextIOEnv(episodes_root=episodes_root, session_id=session_id)

if __name__ == "__main__":
    env = make_text_env()
    env.reset()
    env.begin_turn("سلام")
    out = env.deliver(
        intent="greeting",
        action={"kind": "skill", "name": "reply_greeting", "args": {}},
        text_out="سلام! خوش اومدی.",
        meta={"confidence": 0.9, "u": 0.1, "r_total": 0.6},
        feedback=+1,
    )
    print("env ok — r_ext:", out.r_ext)
