# -*- coding: utf-8 -*-
"""NOEMA • skills/reply_smalltalk.py — Lightweight smalltalk fallback."""

from __future__ import annotations

from typing import Dict, Any


def run(user_text: str = "", **kwargs) -> Dict[str, Any]:
    reply = (
        "Right now I can greet and do simple calculations, and I'm learning the rest. "
        "If you share your goal or a quick example, I can help better."
    )
    return {
        "intent": "smalltalk",
        "outcome": {},
        "text_out": reply,
        "meta": {"confidence": 0.70, "u": 0.20, "risk": 0.0, "r_total": 0.0},
        "extras": {},
        "label_ok": True,
    }
