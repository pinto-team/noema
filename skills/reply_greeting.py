# -*- coding: utf-8 -*-
"""NOEMA • skills/reply_greeting.py — Simple greeting reply (V0)."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from lang.format import load_style, format_reply, Style  # type: ignore
except Exception:
    # Minimal fallback if lang is unavailable
    class Style:
        def __init__(self, tone: str = "friendly", formal: bool = False, prefix_emoji: bool = False) -> None:
            self.tone = tone
            self.formal = formal
            self.prefix_emoji = prefix_emoji
            self.max_len = 500
            self.show_confidence = False

    def load_style(path: str = "config/meta.yaml") -> "Style":  # type: ignore
        return Style()

    def format_reply(*, intent: str, outcome: Dict[str, Any] | None = None, style: Optional["Style"] = None, meta: Optional[Dict[str, Any]] = None) -> str:  # type: ignore
        style = style or Style()
        return "Hello." if style.formal else "Hello! Welcome."


_DEFAULT_META = {
    "confidence": 0.90,
    "u": 0.10,
    "r_total": 0.0,
    "risk": 0.0,
}

def run(
    user_text: str = "",
    *,
    plan: Optional[Dict[str, Any]] = None,
    style: Optional[Style] = None,
    extras: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Produce a short, polite greeting."""
    style = style or load_style()
    outcome = {"variant": "default"}
    meta = dict(_DEFAULT_META)

    # Optional tone control via plan
    tone = (plan or {}).get("tone")
    if isinstance(tone, str):
        t = tone.strip().lower()
        if t in ("formal", "official"):
            style.formal = True
            style.prefix_emoji = False
        elif t in ("friendly", "casual"):
            style.formal = False

    text_out = format_reply(intent="greeting", outcome=outcome, style=style, meta=meta)

    return {
        "intent": "greeting",
        "outcome": outcome,
        "text_out": text_out,
        "meta": meta,
        "extras": dict(extras or {}),
    }

if __name__ == "__main__":
    print(run()["text_out"])            # default
    print(run(plan={"tone": "formal"})["text_out"])  # formal
